resource "kubernetes_manifest"
"cluster-autoscaler_service_account" {
	manifest = {
		"apiVersion" = "v1"
		"kind" = "ServiceAccount"
		"metadata" = {
			"name" =
			var.cluster_autoscaler_name "namespace" =
			var.cluster_autoscaler_namespace "annotations" = {
				"eks.amazonaws.com/role-arn" =
				var.cluster_iam_role_arn
			}
			"labels" = {
				"k8s-addon": var.cluster_k8s_addon "k8s-app": var.cluster_autoscaler_name
			}
		}
	}
}
resource "kubernetes_manifest"
"clusterrole_cluster_autoscaler" {
	manifest = {
		"apiVersion" = "rbac.authorization.k8s.io/v1"
		"kind" = "ClusterRole"
		"metadata" = {
			"labels" = {
				"k8s-addon" =
				var.cluster_k8s_addon "k8s-app" =
				var.cluster_autoscaler_name
			}
			"name" =
			var.cluster_autoscaler_name
		}
		"rules" = [{
			"apiGroups" = ["", ]
			"resources" = ["events", "endpoints", ]
			"verbs" = ["create", "patch", ]
		}, {
			"apiGroups" = ["", ]
			"resources" = ["pods/eviction", ]
			"verbs" = ["create", ]
		}, {
			"apiGroups" = ["", ]
			"resources" = ["pods/status", ]
			"verbs" = ["update", ]
		}, {
			"apiGroups" = ["", ]
			"resourceNames" = [var.cluster_autoscaler_name, ]
			"resources" = ["endpoints", ]
			"verbs" = ["get", "update", ]
		}, {
			"apiGroups" = ["", ]
			"resources" = ["nodes", ]
			"verbs" = ["watch", "list", "get", "update", ]
		}, {
			"apiGroups" = ["", ]
			"resources" = ["namespaces", "pods", "services", "replicationcontrollers", "persistentvolumeclaims", "persistentvolumes", ]
			"verbs" = ["watch", "list", "get", ]
		}, {
			"apiGroups" = ["extensions", ]
			"resources" = ["replicasets", "daemonsets", ]
			"verbs" = ["watch", "list", "get", ]
		}, {
			"apiGroups" = ["policy", ]
			"resources" = ["poddisruptionbudgets", ]
			"verbs" = ["watch", "list", ]
		}, {
			"apiGroups" = ["apps", ]
			"resources" = ["statefulsets", "replicasets", "daemonsets", ]
			"verbs" = ["watch", "list", "get", ]
		}, {
			"apiGroups" = ["storage.k8s.io", ]
			"resources" = ["storageclasses", "csinodes", "csidrivers", "csistoragecapacities", ]
			"verbs" = ["watch", "list", "get", ]
		}, {
			"apiGroups" = ["batch", "extensions", ]
			"resources" = ["jobs", ]
			"verbs" = ["get", "list", "watch", "patch", ]
		}, {
			"apiGroups" = ["coordination.k8s.io", ]
			"resources" = ["leases", ]
			"verbs" = ["create", ]
		}, {
			"apiGroups" = ["coordination.k8s.io", ]
			"resourceNames" = [var.cluster_autoscaler_name, ]
			"resources" = ["leases", ]
			"verbs" = ["get", "update", ]
		}, ]
	}
}
resource "kubernetes_manifest"
"role_kube_system_cluster_autoscaler" {
	manifest = {
		"apiVersion" = "rbac.authorization.k8s.io/v1"
		"kind" = "Role"
		"metadata" = {
			"labels" = {
				"k8s-addon" =
				var.cluster_k8s_addon "k8s-app" =
				var.cluster_autoscaler_name
			}
			"name" =
			var.cluster_autoscaler_name "namespace" =
			var.cluster_autoscaler_namespace
		}
		"rules" = [{
			"apiGroups" = ["", ]
			"resources" = ["configmaps", ]
			"verbs" = ["create", "list", "watch", ]
		}, {
			"apiGroups" = ["", ]
			"resourceNames" = ["cluster-autoscaler-status", "cluster-autoscaler-priority-expander", ]
			"resources" = ["configmaps", ]
			"verbs" = ["delete", "get", "update", "watch", ]
		}, ]
	}
}
resource "kubernetes_manifest"
"clusterrolebinding_cluster_autoscaler" {
	manifest = {
		"apiVersion" = "rbac.authorization.k8s.io/v1"
		"kind" = "ClusterRoleBinding"
		"metadata" = {
			"labels" = {
				"k8s-addon" =
				var.cluster_k8s_addon "k8s-app" =
				var.cluster_autoscaler_name
			}
			"name" =
			var.cluster_autoscaler_name
		}
		"roleRef" = {
			"apiGroup" = "rbac.authorization.k8s.io"
			"kind" = "ClusterRole"
			"name" =
			var.cluster_autoscaler_name
		}
		"subjects" = [{
			"kind" = "ServiceAccount"
			"name" =
			var.cluster_autoscaler_name "namespace" =
			var.cluster_autoscaler_namespace
		}, ]
	}
}
resource "kubernetes_manifest"
"rolebinding_kube_system_cluster_autoscaler" {
	manifest = {
		"apiVersion" = "rbac.authorization.k8s.io/v1"
		"kind" = "RoleBinding"
		"metadata" = {
			"labels" = {
				"k8s-addon" =
				var.cluster_k8s_addon "k8s-app" =
				var.cluster_autoscaler_name
			}
			"name" =
			var.cluster_autoscaler_name "namespace" =
			var.cluster_autoscaler_namespace
		}
		"roleRef" = {
			"apiGroup" = "rbac.authorization.k8s.io"
			"kind" = "Role"
			"name" =
			var.cluster_autoscaler_name
		}
		"subjects" = [{
			"kind" = "ServiceAccount"
			"name" =
			var.cluster_autoscaler_name "namespace" =
			var.cluster_autoscaler_namespace
		}, ]
	}
}
resource "kubernetes_manifest"
"deployment_kube_system_cluster_autoscaler" {
	computed_fields = ["metadata.labels", "metadata.annotations"] manifest = {
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"labels": {
				"app": var.cluster_autoscaler_name
			},
			"name": var.cluster_autoscaler_name,
			"namespace": var.cluster_autoscaler_namespace
		},
		"spec": {
			"replicas": var.replicas,
			"selector": {
				"matchLabels": {
					"app": var.cluster_autoscaler_name
				}
			},
			"template": {
				"metadata": {
					"labels": {
						"app": var.cluster_autoscaler_name
					}
				},
				"spec": {
					"containers": [{
						"command": ["./cluster-autoscaler", " - v=4", " - stderrthreshold=info", " - cloud-provider=aws", " - skip-nodes-with-local-storage=false", " - expander=least-waste", " - node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/${var.eks_cluster_name}", " - balance-similar-node-groups", " - skip-nodes-with-system-pods=false", " - scale-down-delay-after-add=${var.scale_down_delay_after_add}", " - scale-down-delay-after-delete=${var.scale_down_delay_after_delete}", " - scale-down-delay-after-failure=${var.scale_down_delay_after_failure}", " - scan-interval=${var.scan_interval}"],
						"image": var.cluster_docker_image_url,
						"env": [{
							"name": "http_proxy",
							"value": var.http_proxy
						}, {
							"name": "https_proxy",
							"value": var.https_proxy
						}, {
							"name": "no_proxy",
							"value": var.no_proxy
						}],
						"imagePullPolicy": "Always",
						"name": var.cluster_autoscaler_name,
						"resources": {
							"limits": {
								"cpu": var.cluster_cpu_limit,
								"memory": var.cluster_memory_limit
							},
							"requests": {
								"cpu": var.cluster_cpu_request,
								"memory": var.cluster_memory_request
							}
						},
						"volumeMounts": [{
							"mountPath": "/etc/ssl/certs/ca-certificates.crt",
							"name": "ssl-certs",
							"readOnly": true
						}, ]
					}, ],
					"priorityClassName": "system-cluster-critical",
					"securityContext": {
						"fsGroup": 65534,
						"runAsNonRoot": true,
						"runAsUser": 65534
					},
					"serviceAccountName": var.cluster_autoscaler_name,
					"volumes": [{
						"hostPath": {
							"path": "/etc/ssl/certs/ca-bundle.crt"
						},
						"name": "ssl-certs"
					}, ]
				}
			}
		}
	}
}
