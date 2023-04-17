variable "eks_cluster_name" {
  type        = string
  description = "The EKS cluster where this autoscaler will be provisioned"
}

variable "permissions_boundary" {
  type        = string
  description = "ARN of the policy that is used to set the permissions boundary for the autoscaling role"
}

variable "tags" {
  type        = map(string)
  description = "tags supplied from upstream to merge with generated tags"
  default     = {}
}

variable "cluster_autoscaler_name" {
  type        = string
  default     = "cluster-autoscaler"
  description = "The name of the cluster autoscaler"
}

variable "cluster_autoscaler_namespace" {
  type        = string
  default     = "kube-system"
  description = "The namespace of the cluster autoscaler"
}

variable "cluster_iam_role_arn" {
  type        = string
  description = "The arn of the cluster iam role"
}

variable "cluster_k8s_addon" {
  type        = string
  default     = "cluster-autoscaler.addons.k8s.io"
  description = "The k8s addon for the cluster"
}

variable "cluster_cpu_limit" {
  type        = string
  default     = "100"
  description = "The cpu limit for the cluster"
}

variable "cluster_cpu_request" {
  type        = string
  default     = "100"
  description = "The cpu request for the cluster"
}
variable "cluster_memory_limit" {
  type        = string
  default     = "600Mi"
  description = "The memory limit for the cluster"
}

variable "cluster_memory_request" {
  type        = string
  default     = "600Mi"
  description = "The memory request for the cluster"
}

variable "cluster_docker_image_url" {
  type        = string
  default     = "875250343506.dkr.ecr.ap-southeast-2.amazonaws.com/cluster-autoscaler:v1.22.2"
  description = "The docker image url for cluster autoscaler"
}

variable "replicas" {
  type        = string
  default     = "1"
  description = "The number of replicas for cluster autoscaler"
}

variable "http_proxy" {
  type        = string
  default     = "SET_HTTP_PROXY"
  description = "http proxy value"
}
variable "https_proxy" {
  type        = string
  default     = "SET_HTTPS_PROXY"
  description = "https proxy value"
}
variable "no_proxy" {
  type        = string
  default     = "SET_NO_PROXY"
  description = "no proxy value"
}

variable "scale_down_delay_after_add" {
  type        = string
  default     = "10m"
  description = "This will make the cluster autoscaler more responsive to changes in the load. How long after scale up that scale down evaluation resumes?"
}

variable "scale_down_delay_after_delete" {
  type        = string
  default     = "10s"
  description = "This will make the cluster autoscaler more responsive to changes in the load. How long after node deletion that scale down evaluation resumes, defaults to scan-interval"
}

variable "scale_down_delay_after_failure" {
  type        = string
  default     = "3m"
  description = "This will make the cluster autoscaler more responsive to changes in the load. How long after scale down failure that scale down evaluation resumes?"
}

variable "scan_interval" {
  type        = string
  default     = "10s"
  description = "This flag sets the interval between consecutive scans of the cluster state. Decreasing the scan interval will make the cluster autoscaler check the cluster more frequently and react faster to changes in resource usage."
}
