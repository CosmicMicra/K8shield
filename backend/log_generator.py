"""Synthetic Kubernetes API audit log generator.

Produces ``audit.k8s.io/v1`` ``Event`` objects, the same JSON the kube-apiserver
writes through its log backend. Output is JSON Lines, one event per line.

The simulated cluster has two kinds of traffic.

Normal traffic
    Kubelets, control-plane controllers, health probes, in-cluster service
    accounts (cert-manager, Prometheus, CoreDNS, Argo CD, application pods),
    CI/CD deployers and human operators. Some of it is noisy on purpose,
    because real clusters are noisy: CI bursts, platform admins editing
    RoleBindings, developers exec-ing into dev pods, cluster-wide monitoring,
    and the odd 403 from a misconfigured service account.

Injected anomalies (attack campaigns)
    * ``rbac_escalation``: reconnaissance followed by attempts to bind
      ``cluster-admin`` through ClusterRoleBindings.
    * ``exec_anomaly``: ``pods/exec`` into production pods from unfamiliar
      external IP addresses.
    * ``secret_burst``: high-frequency GET/LIST bursts against
      ``/api/v1/namespaces/kube-system/secrets``.

Each injected event carries a ground-truth annotation
(``k8shield.io/injected-anomaly``) so the pipeline can be evaluated. The
feature extractor only reads it to build labels; it is never a model input.
Real audit logs do not have this annotation.

``anomaly_ratio`` is the target fraction of anomalous *entity windows*, where
an entity is a (source IP, user) pair and a window is ``window_minutes`` long.
Detection also runs at that granularity.
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

AUDIT_API_VERSION = "audit.k8s.io/v1"
GROUND_TRUTH_ANNOTATION = "k8shield.io/injected-anomaly"
ANOMALY_TYPES: tuple[str, ...] = ("rbac_escalation", "exec_anomaly", "secret_burst")

# Core API resources have an empty API group; everything else maps to its group.
_API_GROUPS: dict[str, str] = {
    "deployments": "apps",
    "replicasets": "apps",
    "statefulsets": "apps",
    "daemonsets": "apps",
    "leases": "coordination.k8s.io",
    "clusterroles": "rbac.authorization.k8s.io",
    "clusterrolebindings": "rbac.authorization.k8s.io",
    "roles": "rbac.authorization.k8s.io",
    "rolebindings": "rbac.authorization.k8s.io",
    "selfsubjectrulesreviews": "authorization.k8s.io",
    "selfsubjectaccessreviews": "authorization.k8s.io",
    "certificates": "cert-manager.io",
    "applications": "argoproj.io",
    "endpointslices": "discovery.k8s.io",
    "ingresses": "networking.k8s.io",
}
_CLUSTER_SCOPED: frozenset[str] = frozenset(
    {
        "nodes",
        "namespaces",
        "clusterroles",
        "clusterrolebindings",
        "selfsubjectrulesreviews",
        "selfsubjectaccessreviews",
        "persistentvolumes",
    }
)
_SUCCESS_CODE: dict[str, int] = {
    "get": 200,
    "list": 200,
    "watch": 200,
    "create": 201,
    "update": 200,
    "patch": 200,
    "delete": 200,
}

APP_NAMESPACES: tuple[str, ...] = (
    "production",
    "staging",
    "dev",
    "payments",
    "checkout",
    "search",
    "analytics",
    "frontend",
)
SYSTEM_NAMESPACES: tuple[str, ...] = (
    "kube-system",
    "monitoring",
    "cert-manager",
    "argocd",
    "ingress-nginx",
    "logging",
)
ALL_NAMESPACES: tuple[str, ...] = APP_NAMESPACES + SYSTEM_NAMESPACES


# --------------------------------------------------------------------------- #
# Configuration and actor model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Settings for one synthetic audit log stream."""

    duration_minutes: int = 24 * 60
    window_minutes: int = 5
    anomaly_ratio: float = 0.05
    stealth_ratio: float = 0.3  # share of attack campaigns that try to blend in
    seed: int = 42
    start_time: datetime = field(
        default_factory=lambda: datetime(2026, 1, 5, 0, 0, tzinfo=UTC)
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.anomaly_ratio < 1.0:
            raise ValueError(f"anomaly_ratio must be in [0, 1), got {self.anomaly_ratio}")
        if not 0.0 <= self.stealth_ratio <= 1.0:
            raise ValueError(f"stealth_ratio must be in [0, 1], got {self.stealth_ratio}")
        if self.window_minutes <= 0 or self.duration_minutes < self.window_minutes:
            raise ValueError("duration_minutes must be >= window_minutes > 0")
        if self.start_time.tzinfo is None:
            raise ValueError("start_time must be timezone-aware")


@dataclass(frozen=True, slots=True)
class Operation:
    """A weighted API call template that an actor can issue."""

    verb: str
    resource: str
    namespaces: tuple[str, ...] = ()
    subresource: str | None = None
    weight: float = 1.0
    error_rate: float = 0.0
    error_code: int = 403
    request_object: dict[str, Any] | None = None
    non_resource_url: str | None = None


@dataclass(frozen=True, slots=True)
class Actor:
    """A recurring identity in the simulated cluster."""

    username: str
    groups: tuple[str, ...]
    source_ips: tuple[str, ...]
    user_agents: tuple[str, ...]
    operations: tuple[Operation, ...]
    activity_prob: float  # probability of being active in a given window
    mean_requests: float  # Poisson mean of requests per active window
    burst_prob: float = 0.0  # probability an active window is a burst
    burst_multiplier: float = 1.0


def _sa(namespace: str, name: str) -> tuple[str, tuple[str, ...]]:
    return (
        f"system:serviceaccount:{namespace}:{name}",
        (
            "system:serviceaccounts",
            f"system:serviceaccounts:{namespace}",
            "system:authenticated",
        ),
    )


def _human(name: str, *groups: str) -> tuple[str, tuple[str, ...]]:
    return f"{name}@k8shield.corp", (*groups, "system:authenticated")


KUBECTL_MAC = "kubectl/v1.29.3 (darwin/arm64) kubernetes/6813625"
KUBECTL_LINUX = "kubectl/v1.29.3 (linux/amd64) kubernetes/6813625"
KUBELET_UA = "kubelet/v1.29.4 (linux/amd64) kubernetes/55019c8"


def build_normal_actors() -> list[Actor]:
    """Return the benign identities that make up the cluster's baseline traffic."""
    actors: list[Actor] = []

    # Kubelets: steady node heartbeats and pod status updates.
    for i in range(1, 7):
        actors.append(
            Actor(
                username=f"system:node:ip-10-0-1-{10 + i}",
                groups=("system:nodes", "system:authenticated"),
                source_ips=(f"10.0.1.{10 + i}",),
                user_agents=(KUBELET_UA,),
                operations=(
                    Operation("patch", "nodes", subresource="status", weight=4),
                    Operation("get", "nodes", weight=2),
                    Operation("update", "leases", ("kube-node-lease",), weight=4),
                    Operation("patch", "pods", APP_NAMESPACES, subresource="status", weight=3),
                    Operation("get", "configmaps", APP_NAMESPACES, weight=1),
                    Operation("get", "secrets", APP_NAMESPACES, weight=1),
                ),
                activity_prob=1.0,
                mean_requests=22,
            )
        )

    # Load-balancer and kube-probe health checks against non-resource URLs.
    for ip in ("10.0.0.2", "10.0.0.3"):
        actors.append(
            Actor(
                username="system:anonymous",
                groups=("system:unauthenticated",),
                source_ips=(ip,),
                user_agents=("kube-probe/1.29",),
                operations=(
                    Operation("get", "", non_resource_url="/readyz", weight=3),
                    Operation("get", "", non_resource_url="/livez", weight=3),
                    Operation("get", "", non_resource_url="/healthz", weight=2),
                ),
                activity_prob=1.0,
                mean_requests=10,
            )
        )

    # Control-plane components.
    actors.append(
        Actor(
            username="system:kube-controller-manager",
            groups=("system:authenticated",),
            source_ips=("10.0.0.10",),
            user_agents=("kube-controller-manager/v1.29.4 (linux/amd64) kubernetes/55019c8",),
            operations=(
                Operation("update", "leases", ("kube-system",), weight=4),
                Operation("list", "pods", APP_NAMESPACES, weight=2),
                Operation("create", "pods", APP_NAMESPACES, weight=1),
                Operation("update", "replicasets", APP_NAMESPACES, weight=2),
                Operation("get", "serviceaccounts", ALL_NAMESPACES, weight=1),
            ),
            activity_prob=1.0,
            mean_requests=18,
        )
    )
    actors.append(
        Actor(
            username="system:kube-scheduler",
            groups=("system:authenticated",),
            source_ips=("10.0.0.10",),
            user_agents=("kube-scheduler/v1.29.4 (linux/amd64) kubernetes/55019c8",),
            operations=(
                Operation("update", "leases", ("kube-system",), weight=4),
                Operation("create", "pods", APP_NAMESPACES, subresource="binding", weight=2),
                Operation("watch", "pods", weight=1),
            ),
            activity_prob=1.0,
            mean_requests=9,
        )
    )

    # cert-manager reads and renews TLS secrets across many namespaces.
    name, groups = _sa("cert-manager", "cert-manager")
    actors.append(
        Actor(
            username=name,
            groups=groups,
            source_ips=("10.244.3.17",),
            user_agents=("cert-manager/v1.14.4 (linux/amd64) cert-manager/a3d5c2e",),
            operations=(
                Operation("get", "secrets", ALL_NAMESPACES, weight=4),
                Operation("list", "certificates", APP_NAMESPACES, weight=2),
                Operation("update", "secrets", APP_NAMESPACES, weight=1),
                Operation("update", "leases", ("kube-system",), weight=2),
            ),
            activity_prob=0.9,
            mean_requests=12,
        )
    )

    # Prometheus discovers scrape targets cluster-wide.
    name, groups = _sa("monitoring", "prometheus-k8s")
    actors.append(
        Actor(
            username=name,
            groups=groups,
            source_ips=("10.244.1.42",),
            user_agents=("Prometheus/2.51.1",),
            operations=(
                Operation("list", "pods", ALL_NAMESPACES, weight=3),
                Operation("list", "endpoints", ALL_NAMESPACES, weight=3),
                Operation("watch", "services", ALL_NAMESPACES, weight=2),
                Operation("get", "nodes", subresource="metrics", weight=1),
            ),
            activity_prob=1.0,
            mean_requests=14,
        )
    )

    # CoreDNS watches services and endpoint slices.
    name, groups = _sa("kube-system", "coredns")
    actors.append(
        Actor(
            username=name,
            groups=groups,
            source_ips=("10.244.0.5",),
            user_agents=("coredns/v1.11.1 (linux/amd64)",),
            operations=(
                Operation("watch", "endpointslices", weight=2),
                Operation("watch", "services", weight=2),
                Operation("list", "namespaces", weight=1),
            ),
            activity_prob=0.8,
            mean_requests=5,
        )
    )

    # Argo CD reconciles applications and occasionally runs a large sync burst.
    name, groups = _sa("argocd", "argocd-application-controller")
    actors.append(
        Actor(
            username=name,
            groups=groups,
            source_ips=("10.244.2.8",),
            user_agents=("argocd-application-controller/v2.10.6",),
            operations=(
                Operation("get", "deployments", APP_NAMESPACES, weight=3),
                Operation("patch", "deployments", APP_NAMESPACES, weight=1),
                Operation("list", "applications", ("argocd",), weight=2),
                Operation("get", "configmaps", APP_NAMESPACES, weight=1),
                Operation("get", "services", APP_NAMESPACES, weight=1),
            ),
            activity_prob=0.95,
            mean_requests=16,
            burst_prob=0.06,
            burst_multiplier=8.0,
        )
    )

    # Application workloads read their own config and secrets, with a few
    # 403s from an under-privileged service account.
    for ns, app, ip in (
        ("payments", "payments-api", "10.244.4.21"),
        ("checkout", "checkout-svc", "10.244.4.33"),
        ("search", "search-indexer", "10.244.5.12"),
        ("production", "web-backend", "10.244.5.40"),
    ):
        name, groups = _sa(ns, app)
        actors.append(
            Actor(
                username=name,
                groups=groups,
                source_ips=(ip,),
                user_agents=("Go-http-client/2.0", f"{app}/1.8.2"),
                operations=(
                    Operation("get", "secrets", (ns,), weight=2, error_rate=0.04),
                    Operation("get", "configmaps", (ns,), weight=2),
                    Operation("list", "pods", (ns,), weight=1, error_rate=0.08),
                    Operation("get", "leases", (ns,), weight=1),
                ),
                activity_prob=0.55,
                mean_requests=6,
            )
        )

    # Fluent Bit enriches log lines with pod metadata.
    name, groups = _sa("logging", "fluent-bit")
    actors.append(
        Actor(
            username=name,
            groups=groups,
            source_ips=("10.244.6.3",),
            user_agents=("fluent-bit/3.0.2",),
            operations=(
                Operation("get", "pods", APP_NAMESPACES, weight=4),
                Operation("list", "namespaces", weight=1),
            ),
            activity_prob=0.7,
            mean_requests=8,
        )
    )

    # In-cluster GitLab runner: rare but large Helm deploy bursts.
    name, groups = _sa("ci", "gitlab-runner")
    actors.append(
        Actor(
            username=name,
            groups=groups,
            source_ips=("10.244.7.9",),
            user_agents=("Helm/3.14.3",),
            operations=(
                Operation("get", "secrets", APP_NAMESPACES, weight=2),
                Operation("create", "secrets", APP_NAMESPACES, weight=1),
                Operation("patch", "deployments", APP_NAMESPACES, weight=3),
                Operation("create", "services", APP_NAMESPACES, weight=1),
                Operation("list", "pods", APP_NAMESPACES, weight=2),
                Operation("create", "configmaps", APP_NAMESPACES, weight=1, error_rate=0.05, error_code=409),
            ),
            activity_prob=0.12,
            mean_requests=40,
            burst_prob=0.35,
            burst_multiplier=4.0,
        )
    )

    # GitHub Actions deployer: legitimate traffic from public cloud IPs.
    name, groups = _sa("ci", "github-deployer")
    actors.append(
        Actor(
            username=name,
            groups=groups,
            source_ips=("20.62.134.17", "20.85.12.201", "52.234.40.9"),
            user_agents=(KUBECTL_LINUX, "Helm/3.14.3"),
            operations=(
                Operation("patch", "deployments", ("frontend", "search", "analytics"), weight=3),
                Operation("get", "deployments", ("frontend", "search", "analytics"), weight=3),
                Operation("list", "pods", ("frontend", "search", "analytics"), weight=2),
            ),
            activity_prob=0.06,
            mean_requests=25,
        )
    )

    # Platform admin: occasionally manages RBAC as part of their job.
    name, groups = _human("alice", "platform-admins")
    actors.append(
        Actor(
            username=name,
            groups=groups,
            source_ips=("192.168.10.21", "192.168.10.22"),
            user_agents=(KUBECTL_MAC, "k9s/v0.32.4"),
            operations=(
                Operation("get", "pods", ALL_NAMESPACES, weight=4),
                Operation("list", "deployments", ALL_NAMESPACES, weight=3),
                Operation("get", "nodes", weight=2),
                Operation(
                    "create",
                    "rolebindings",
                    APP_NAMESPACES,
                    weight=0.4,
                    request_object={
                        "kind": "RoleBinding",
                        "roleRef": {"kind": "ClusterRole", "name": "edit"},
                    },
                ),
                Operation("patch", "roles", APP_NAMESPACES, weight=0.3),
                Operation("delete", "pods", APP_NAMESPACES, weight=0.5),
            ),
            activity_prob=0.30,
            mean_requests=14,
        )
    )

    # Developers: read workloads, tail logs, exec into dev/staging pods, and
    # sometimes get 403 when they wander into production.
    for dev, ips in (
        ("bob", ("192.168.10.45",)),
        ("carol", ("192.168.10.61", "73.92.18.144")),  # carol also works from home
        ("dmitri", ("192.168.10.77",)),
        ("erin", ("192.168.11.12",)),
    ):
        name, groups = _human(dev, "developers")
        actors.append(
            Actor(
                username=name,
                groups=groups,
                source_ips=ips,
                user_agents=(KUBECTL_MAC, "Lens/2024.3.1", "k9s/v0.32.4"),
                operations=(
                    Operation("get", "pods", ("dev", "staging", "frontend", "search"), weight=5),
                    Operation("list", "pods", ("dev", "staging"), weight=3),
                    Operation("get", "pods", ("dev", "staging"), subresource="log", weight=3),
                    Operation("create", "pods", ("dev", "staging"), subresource="exec", weight=0.6),
                    Operation("get", "pods", ("production",), weight=0.5, error_rate=0.6),
                    Operation("get", "configmaps", ("dev", "staging"), weight=1),
                ),
                activity_prob=0.35,
                mean_requests=11,
            )
        )

    actors.extend(build_rare_benign_actors())
    return actors


def build_rare_benign_actors() -> list[Actor]:
    """Rare but legitimate behaviour that resembles an attack.

    These identities are a large source of false positives in real clusters,
    for static rules and for learned models alike. Leaving them out makes any
    detector look unrealistically good.
    """
    actors: list[Actor] = []

    # On-call SRE responding to an incident from home: execs into production
    # pods from a residential IP with a newer kubectl, and hits a few 403s.
    name, groups = _human("frank", "sre-oncall")
    actors.append(
        Actor(
            username=name,
            groups=groups,
            source_ips=("98.207.33.12", "192.168.10.90"),
            user_agents=("kubectl/v1.30.1 (darwin/arm64) kubernetes/6911225",),
            operations=(
                Operation("get", "pods", ("production", "payments", "checkout"), weight=4),
                Operation("get", "pods", ("production", "payments"), subresource="log", weight=3),
                Operation("create", "pods", ("production", "payments"), subresource="exec", weight=1.5,
                          error_rate=0.15),
                Operation("list", "events", ALL_NAMESPACES, weight=2),
                Operation("get", "secrets", ("production",), weight=0.5, error_rate=0.5),
            ),
            activity_prob=0.012,
            mean_requests=18,
        )
    )

    # Token-rotation fallout: a workload keeps retrying with a revoked token.
    name, groups = _sa("payments", "payments-api")
    actors.append(
        Actor(
            username=name,
            groups=groups,
            source_ips=("10.244.4.21",),
            user_agents=("Go-http-client/2.0",),
            operations=(
                Operation("get", "secrets", ("payments",), weight=3, error_rate=0.9, error_code=401),
                Operation("get", "configmaps", ("payments",), weight=1, error_rate=0.9, error_code=401),
            ),
            activity_prob=0.01,
            mean_requests=60,
        )
    )

    # Velero backups: roughly hourly sweep of secrets and config in every namespace.
    name, groups = _sa("velero", "velero")
    actors.append(
        Actor(
            username=name,
            groups=groups,
            source_ips=("10.244.6.50",),
            user_agents=("velero/v1.13.2 (linux/amd64) 8e5ee5a",),
            operations=(
                Operation("list", "secrets", ALL_NAMESPACES, weight=2),
                Operation("list", "configmaps", ALL_NAMESPACES, weight=2),
                Operation("list", "persistentvolumeclaims", ALL_NAMESPACES, weight=1),
            ),
            activity_prob=1 / 12,
            mean_requests=45,
        )
    )

    # New contractor exploring the cluster: rare client and plenty of 403s.
    name, groups = _human("grace", "contractors")
    actors.append(
        Actor(
            username=name,
            groups=groups,
            source_ips=("192.168.12.8",),
            user_agents=("Lens/2024.3.1", "kubectl/v1.28.2 (windows/amd64) kubernetes/89a4ea3"),
            operations=(
                Operation("list", "pods", ALL_NAMESPACES, weight=3, error_rate=0.3),
                Operation("get", "deployments", APP_NAMESPACES, weight=2, error_rate=0.3),
                Operation("list", "secrets", APP_NAMESPACES, weight=0.5, error_rate=0.8),
            ),
            activity_prob=0.03,
            mean_requests=12,
        )
    )
    return actors


# --------------------------------------------------------------------------- #
# Event construction
# --------------------------------------------------------------------------- #
def _iso(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def build_audit_event(
    *,
    ts: datetime,
    username: str,
    groups: Sequence[str],
    source_ip: str,
    user_agent: str,
    verb: str,
    resource: str,
    namespace: str | None,
    name: str | None,
    subresource: str | None,
    code: int,
    request_object: dict[str, Any] | None = None,
    non_resource_url: str | None = None,
    anomaly_type: str | None = None,
    event_uuid: str | None = None,
    latency_ms: float = 8.0,
) -> dict[str, Any]:
    """Build one ``audit.k8s.io/v1`` Event dictionary."""
    if non_resource_url is not None:
        request_uri = non_resource_url
        object_ref = None
    else:
        group = _API_GROUPS.get(resource, "")
        prefix = f"/apis/{group}/v1" if group else "/api/v1"
        parts = [prefix]
        if namespace and resource not in _CLUSTER_SCOPED:
            parts += ["namespaces", namespace]
        parts.append(resource)
        if name:
            parts.append(name)
        if subresource:
            parts.append(subresource)
        request_uri = "/".join(parts)
        if verb == "watch":
            request_uri += "?watch=true"
        object_ref = {"resource": resource, "apiVersion": "v1"}
        if group:
            object_ref["apiGroup"] = group
        if namespace and resource not in _CLUSTER_SCOPED:
            object_ref["namespace"] = namespace
        if name:
            object_ref["name"] = name
        if subresource:
            object_ref["subresource"] = subresource

    allowed = code not in (401, 403)
    annotations: dict[str, str] = {
        "authorization.k8s.io/decision": "allow" if allowed else "forbid",
        "authorization.k8s.io/reason": (
            "RBAC: allowed by ClusterRoleBinding" if allowed else ""
        ),
    }
    if anomaly_type:
        annotations[GROUND_TRUTH_ANNOTATION] = anomaly_type

    event: dict[str, Any] = {
        "kind": "Event",
        "apiVersion": AUDIT_API_VERSION,
        "level": "Request" if request_object else "Metadata",
        "auditID": event_uuid or str(uuid.uuid4()),
        "stage": "ResponseComplete",
        "requestURI": request_uri,
        "verb": verb,
        "user": {"username": username, "groups": list(groups)},
        "sourceIPs": [source_ip],
        "userAgent": user_agent,
        "responseStatus": {"metadata": {}, "code": code},
        "requestReceivedTimestamp": _iso(ts),
        "stageTimestamp": _iso(ts + timedelta(milliseconds=latency_ms)),
        "annotations": annotations,
    }
    if object_ref is not None:
        event["objectRef"] = object_ref
    if request_object:
        event["requestObject"] = request_object
    if not allowed:
        event["responseStatus"].update(
            {"status": "Failure", "reason": "Forbidden" if code == 403 else "Unauthorized"}
        )
    return event


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
class AuditLogGenerator:
    """Generate a time-ordered stream of synthetic audit events.

    Example:
        >>> gen = AuditLogGenerator(GeneratorConfig(duration_minutes=60, seed=7))
        >>> events = list(gen.generate())
        >>> events[0]["apiVersion"]
        'audit.k8s.io/v1'
    """

    _EXTERNAL_ATTACK_PREFIXES = ("185.220.101", "45.155.205", "103.75.190", "91.240.118", "194.26.29")
    _ATTACKER_UAS = (
        "kubectl/v1.18.0 (linux/amd64) kubernetes/9e99141",
        "python-requests/2.31.0",
        "Go-http-client/1.1",
        "curl/7.88.1",
        "kubectl/v1.29.3 (linux/amd64) kubernetes/6813625",  # blends in with normal kubectl
    )

    def __init__(self, config: GeneratorConfig | None = None, actors: list[Actor] | None = None):
        self.config = config or GeneratorConfig()
        self.actors = actors or build_normal_actors()
        self.rng = np.random.default_rng(self.config.seed)
        self.stats: Counter[str] = Counter()

    # ---- helpers --------------------------------------------------------- #
    def _uuid(self) -> str:
        return str(uuid.UUID(bytes=self.rng.bytes(16), version=4))

    def _choice(self, seq: Sequence[Any]) -> Any:
        return seq[int(self.rng.integers(len(seq)))]

    def _rand_ts(self, start: datetime, span_s: float) -> datetime:
        return start + timedelta(seconds=float(self.rng.uniform(0, span_s)))

    def _pod_name(self, ns: str) -> str:
        app = self._choice(("api", "web", "worker", "db", "cache"))
        suffix = "".join(self._choice("bcdfghjklmnpqrstvwxz2456789") for _ in range(5))
        return f"{ns}-{app}-7d9f8c{int(self.rng.integers(10, 99))}-{suffix}"

    def _event_from_op(self, actor: Actor, op: Operation, ts: datetime, ip: str, ua: str) -> dict[str, Any]:
        ns = self._choice(op.namespaces) if op.namespaces else None
        name: str | None = None
        if op.verb in {"get", "patch", "update", "delete"} or op.subresource:
            if op.resource == "pods":
                name = self._pod_name(ns or "default")
            elif op.resource == "nodes":
                name = actor.username.split(":")[-1] if actor.username.startswith("system:node:") else "ip-10-0-1-11"
            elif op.resource == "leases":
                name = actor.username.split(":")[-1]
            elif op.resource:
                name = f"{op.resource.rstrip('s')}-{int(self.rng.integers(1, 40))}"
        failed = op.error_rate > 0 and self.rng.random() < op.error_rate
        code = op.error_code if failed else _SUCCESS_CODE.get(op.verb, 200)
        return build_audit_event(
            ts=ts,
            username=actor.username,
            groups=actor.groups,
            source_ip=ip,
            user_agent=ua,
            verb=op.verb,
            resource=op.resource,
            namespace=ns,
            name=name,
            subresource=op.subresource,
            code=code,
            request_object=op.request_object,
            non_resource_url=op.non_resource_url,
            event_uuid=self._uuid(),
            latency_ms=float(self.rng.gamma(2.0, 6.0)),
        )

    def _normal_window(self, actor: Actor, start: datetime, span_s: float) -> list[dict[str, Any]]:
        mean = actor.mean_requests
        if actor.burst_prob and self.rng.random() < actor.burst_prob:
            mean *= actor.burst_multiplier
        n = max(1, int(self.rng.poisson(mean)))
        weights = np.array([op.weight for op in actor.operations], dtype=float)
        weights /= weights.sum()
        op_idx = self.rng.choice(len(actor.operations), size=n, p=weights)
        # A session usually sticks to one IP and one client.
        ip = self._choice(actor.source_ips)
        ua = self._choice(actor.user_agents)
        return [
            self._event_from_op(actor, actor.operations[i], self._rand_ts(start, span_s), ip, ua)
            for i in op_idx
        ]

    # ---- attack campaigns ------------------------------------------------ #
    def _attacker_identity(self) -> tuple[str, tuple[str, ...], str]:
        """Pick a compromised identity and the IP it is used from."""
        roll = self.rng.random()
        if roll < 0.4:  # stolen developer kubeconfig used from outside
            dev = self._choice(("bob", "carol", "dmitri", "erin"))
            name, groups = _human(dev, "developers")
        elif roll < 0.8:  # compromised workload service account token
            ns = self._choice(("default", "frontend", "analytics", "search"))
            name, groups = _sa(ns, self._choice(("default", "app-sa", "jupyter")))
        else:  # unknown human identity from a third-party OIDC token
            name, groups = _human(f"contractor{int(self.rng.integers(1, 99))}", "contractors")
        if self.rng.random() < 0.7:
            ip = f"{self._choice(self._EXTERNAL_ATTACK_PREFIXES)}.{int(self.rng.integers(2, 254))}"
        else:  # pivot from a compromised pod inside the cluster
            ip = f"10.244.{int(self.rng.integers(8, 30))}.{int(self.rng.integers(2, 254))}"
        return name, groups, ip

    def _attack_events(self, kind: str, start: datetime, span_s: float) -> list[dict[str, Any]]:
        name, groups, ip = self._attacker_identity()
        ua = self._choice(self._ATTACKER_UAS)
        # A share of campaigns are deliberately stealthy: they use the cluster's
        # common kubectl build, pivot from inside the pod network, keep volume
        # low and avoid tripping authorization failures.
        stealthy = self.rng.random() < self.config.stealth_ratio
        if stealthy:
            ua = self._choice((KUBECTL_MAC, KUBECTL_LINUX))
            ip = f"10.244.{int(self.rng.integers(8, 30))}.{int(self.rng.integers(2, 254))}"
        self.stats[f"attacks.stealthy.{kind}"] += int(stealthy)
        events: list[dict[str, Any]] = []

        def emit(verb: str, resource: str, ns: str | None, *, obj_name: str | None = None,
                 sub: str | None = None, code: int | None = None, req: dict[str, Any] | None = None,
                 ts: datetime | None = None) -> None:
            events.append(
                build_audit_event(
                    ts=ts or self._rand_ts(start, span_s),
                    username=name,
                    groups=groups,
                    source_ip=ip,
                    user_agent=ua,
                    verb=verb,
                    resource=resource,
                    namespace=ns,
                    name=obj_name,
                    subresource=sub,
                    code=code if code is not None else _SUCCESS_CODE.get(verb, 200),
                    request_object=req,
                    anomaly_type=kind,
                    event_uuid=self._uuid(),
                    latency_ms=float(self.rng.gamma(2.0, 6.0)),
                )
            )

        if kind == "rbac_escalation":
            # Recon: "kubectl auth can-i --list", then enumerate RBAC objects.
            for _ in range(int(self.rng.integers(1, 4))):
                emit("create", "selfsubjectrulesreviews", None)
            for _ in range(int(self.rng.integers(1, 3) if stealthy else self.rng.integers(2, 8))):
                emit(self._choice(("list", "get")), self._choice(("clusterroles", "clusterrolebindings")), None,
                     code=200 if stealthy else self._choice((200, 403)))
            fail_rate = float(self.rng.uniform(0.0, 0.15) if stealthy else self.rng.uniform(0.2, 1.0))
            for _ in range(int(self.rng.integers(1, 3 if stealthy else 6))):
                binding = f"cluster-admin-{self._choice(('sa', 'debug', 'tmp', 'ops'))}-{int(self.rng.integers(100, 999))}"
                emit("create", "clusterrolebindings", None, obj_name=binding,
                     code=403 if self.rng.random() < fail_rate else 201,
                     req={"kind": "ClusterRoleBinding",
                          "roleRef": {"kind": "ClusterRole", "name": "cluster-admin"},
                          "subjects": [{"kind": "User", "name": name}]})
            if self.rng.random() < 0.5:
                emit("patch", "clusterroles", None, obj_name="system:aggregate-to-edit",
                     code=403 if self.rng.random() < fail_rate else 200)
            if self.rng.random() < 0.5:
                emit("list", "secrets", self._choice(("kube-system", "production")), code=self._choice((200, 403)))

        elif kind == "exec_anomaly":
            targets = ("production", "payments", "checkout")
            n_ns = int(self.rng.integers(1, 4))
            namespaces = tuple(self._choice(targets) for _ in range(n_ns))
            exec_fail = 0.05 if stealthy else 0.3
            for ns in namespaces:
                emit("list", "pods", ns, code=200 if stealthy else self._choice((200, 200, 403)))
            for _ in range(int(self.rng.integers(1, 3) if stealthy else self.rng.integers(2, 10))):
                ns = self._choice(namespaces)
                emit("create", "pods", ns, obj_name=self._pod_name(ns), sub="exec",
                     code=403 if self.rng.random() < exec_fail else 101)
            for _ in range(int(self.rng.integers(0, 2 if stealthy else 6))):
                ns = self._choice(namespaces)
                emit(self._choice(("get", "list")), "secrets", ns,
                     code=200 if stealthy else self._choice((200, 403)))
            if stealthy:  # blend in with ordinary read traffic
                for _ in range(int(self.rng.integers(3, 12))):
                    ns = self._choice(namespaces)
                    emit("get", "pods", ns, obj_name=self._pod_name(ns))

        elif kind == "secret_burst":
            # Bursts are tightly packed in time: up to a few requests per second.
            n = int(self.rng.integers(30, 120) if stealthy else self.rng.integers(60, 600))
            burst_span = float(self.rng.uniform(20, min(180.0, span_s)))
            burst_start = self._rand_ts(start, max(1.0, span_s - burst_span))
            fail_rate = float(self.rng.uniform(0.0, 0.1) if stealthy else self.rng.uniform(0.05, 0.9))
            for _ in range(n):
                verb = "list" if self.rng.random() < 0.3 else "get"
                emit(verb, "secrets", "kube-system",
                     obj_name=None if verb == "list" else f"{self._choice(('bootstrap-token', 'default-token', 'etcd-client', 'admin-token'))}-{int(self.rng.integers(1000, 9999))}",
                     code=403 if self.rng.random() < fail_rate else 200,
                     ts=self._rand_ts(burst_start, burst_span))
        else:
            raise ValueError(f"Unknown anomaly type: {kind!r}")
        return events

    # ---- public API ------------------------------------------------------ #
    def generate(self) -> Iterator[dict[str, Any]]:
        """Yield audit events in timestamp order, one window at a time."""
        cfg = self.config
        span_s = cfg.window_minutes * 60.0
        n_windows = cfg.duration_minutes // cfg.window_minutes
        ratio = cfg.anomaly_ratio
        for w in range(n_windows):
            start = cfg.start_time + timedelta(minutes=w * cfg.window_minutes)
            window_events: list[dict[str, Any]] = []
            active = 0
            for actor in self.actors:
                if self.rng.random() < actor.activity_prob:
                    active += 1
                    window_events.extend(self._normal_window(actor, start, span_s))
            # Expected anomalous groups so that anomalies / (anomalies + normal) ~= ratio.
            n_attacks = int(self.rng.poisson(active * ratio / (1.0 - ratio))) if ratio > 0 else 0
            for _ in range(n_attacks):
                kind = self._choice(ANOMALY_TYPES)
                window_events.extend(self._attack_events(kind, start, span_s))
                self.stats[f"attacks.{kind}"] += 1
            self.stats["windows"] += 1
            self.stats["normal_entity_windows"] += active
            self.stats["anomalous_entity_windows"] += n_attacks
            window_events.sort(key=lambda e: e["requestReceivedTimestamp"])
            for ev in window_events:
                self.stats["events.total"] += 1
                if GROUND_TRUTH_ANNOTATION in ev["annotations"]:
                    self.stats["events.anomalous"] += 1
                yield ev

    def write_jsonl(self, path: str | Path) -> dict[str, int]:
        """Write all events to ``path`` as JSON Lines and return generation stats."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for ev in self.generate():
                fh.write(json.dumps(ev, separators=(",", ":")))
                fh.write("\n")
        stats = dict(self.stats)
        logger.info(
            "Wrote %d events (%d anomalous) across %d windows to %s",
            stats.get("events.total", 0), stats.get("events.anomalous", 0), stats.get("windows", 0), path,
        )
        return stats


def generate_logs(path: str | Path, config: GeneratorConfig | None = None) -> dict[str, int]:
    """Convenience wrapper: generate a log file and return its stats."""
    return AuditLogGenerator(config).write_jsonl(path)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Kubernetes audit logs.")
    parser.add_argument("-o", "--output", default="data/raw/audit.jsonl")
    parser.add_argument("--minutes", type=int, default=24 * 60, help="Simulated duration")
    parser.add_argument("--anomaly-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    stats = generate_logs(
        args.output,
        GeneratorConfig(duration_minutes=args.minutes, anomaly_ratio=args.anomaly_ratio, seed=args.seed),
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    _cli()
