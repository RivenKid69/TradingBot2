{{/*
CCEA Cloud Helm Template Helpers
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "ccea-cloud.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "ccea-cloud.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "ccea-cloud.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "ccea-cloud.labels" -}}
helm.sh/chart: {{ include "ccea-cloud.chart" . }}
{{ include "ccea-cloud.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "ccea-cloud.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ccea-cloud.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Control Plane labels
*/}}
{{- define "ccea-cloud.controlPlane.labels" -}}
{{ include "ccea-cloud.labels" . }}
app.kubernetes.io/component: control-plane
{{- end }}

{{/*
Control Plane selector labels
*/}}
{{- define "ccea-cloud.controlPlane.selectorLabels" -}}
{{ include "ccea-cloud.selectorLabels" . }}
app.kubernetes.io/component: control-plane
{{- end }}

{{/*
Builder labels
*/}}
{{- define "ccea-cloud.builder.labels" -}}
{{ include "ccea-cloud.labels" . }}
app.kubernetes.io/component: builder
{{- end }}

{{/*
Telemetry Ingester labels
*/}}
{{- define "ccea-cloud.telemetry.labels" -}}
{{ include "ccea-cloud.labels" . }}
app.kubernetes.io/component: telemetry-ingester
{{- end }}

{{/*
Governance labels
*/}}
{{- define "ccea-cloud.governance.labels" -}}
{{ include "ccea-cloud.labels" . }}
app.kubernetes.io/component: governance
{{- end }}

{{/*
Create the name of the service account
*/}}
{{- define "ccea-cloud.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "ccea-cloud.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Database URL
*/}}
{{- define "ccea-cloud.databaseUrl" -}}
{{- if .Values.postgresql.enabled }}
postgresql://{{ .Values.postgresql.auth.username }}:$(POSTGRES_PASSWORD)@{{ include "ccea-cloud.fullname" . }}-postgresql:5432/{{ .Values.postgresql.auth.database }}
{{- else }}
{{ .Values.externalDatabase.url }}
{{- end }}
{{- end }}

{{/*
Redis URL
*/}}
{{- define "ccea-cloud.redisUrl" -}}
{{- if .Values.redis.enabled }}
redis://:$(REDIS_PASSWORD)@{{ include "ccea-cloud.fullname" . }}-redis-master:6379/0
{{- else }}
{{ .Values.externalRedis.url }}
{{- end }}
{{- end }}

{{/*
Registry URL
*/}}
{{- define "ccea-cloud.registryUrl" -}}
{{- if .Values.registry.enabled }}
http://{{ include "ccea-cloud.fullname" . }}-registry:5000
{{- else }}
{{ .Values.externalRegistry.url }}
{{- end }}
{{- end }}

{{/*
Image tag
*/}}
{{- define "ccea-cloud.imageTag" -}}
{{- .Values.image.tag | default .Chart.AppVersion }}
{{- end }}

{{/*
Security context for pods
*/}}
{{- define "ccea-cloud.podSecurityContext" -}}
runAsNonRoot: true
runAsUser: 1000
fsGroup: 1000
seccompProfile:
  type: RuntimeDefault
{{- end }}

{{/*
Security context for containers
*/}}
{{- define "ccea-cloud.containerSecurityContext" -}}
readOnlyRootFilesystem: true
allowPrivilegeEscalation: false
capabilities:
  drop:
    - ALL
{{- end }}

{{/*
Common environment variables
*/}}
{{- define "ccea-cloud.commonEnv" -}}
- name: CCEA_ENV
  value: {{ .Values.global.environment | default "production" | quote }}
- name: CCEA_DATA_RESIDENCY
  value: {{ .Values.global.dataResidency | quote }}
- name: CCEA_AIR_GAPPED_MODE
  value: {{ .Values.global.airGapped | quote }}
- name: CCEA_LOG_LEVEL
  value: {{ .Values.controlPlane.env.CCEA_LOG_LEVEL | default "INFO" | quote }}
- name: CCEA_TELEMETRY_REDACTION_MANDATORY
  value: "true"
{{- end }}
