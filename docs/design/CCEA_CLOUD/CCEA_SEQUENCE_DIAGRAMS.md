# CCEA Sequence Diagrams

> **Version**: 2.0.0
> **Date**: 2025-12-16
> **Status**: APPROVED | **All Flows Implemented**

## 1. Agent Enrollment Flow

```
┌──────┐          ┌───────┐          ┌───────┐          ┌────────┐
│ User │          │ Cloud │          │ Agent │          │ Vault  │
└──┬───┘          └───┬───┘          └───┬───┘          └───┬────┘
   │                  │                  │                  │
   │ 1. Request       │                  │                  │
   │ enrollment token │                  │                  │
   │─────────────────▶│                  │                  │
   │                  │                  │                  │
   │ 2. Token (TTL)   │                  │                  │
   │◀─────────────────│                  │                  │
   │                  │                  │                  │
   │ 3. agent enroll  │                  │                  │
   │ --token=<token>  │                  │                  │
   │──────────────────────────────────────▶                 │
   │                  │                  │                  │
   │                  │                  │ 4. Generate      │
   │                  │                  │ device keypair   │
   │                  │                  │─────────────────▶│
   │                  │                  │                  │
   │                  │                  │ 5. Store private │
   │                  │                  │ key locally      │
   │                  │                  │◀─────────────────│
   │                  │                  │                  │
   │                  │ 6. Register      │                  │
   │                  │ public key       │                  │
   │                  │◀─────────────────│                  │
   │                  │                  │                  │
   │                  │ 7. Validate      │                  │
   │                  │ token + store    │                  │
   │                  │ agent record     │                  │
   │                  │                  │                  │
   │                  │ 8. Return        │                  │
   │                  │ agent_id         │                  │
   │                  │─────────────────▶│                  │
   │                  │                  │                  │
   │ 9. Enrollment    │                  │                  │
   │ complete         │                  │                  │
   │◀──────────────────────────────────────                 │
   │                  │                  │                  │
```

**Notes:**
- Token имеет TTL (например, 1 час)
- Private key никогда не покидает Agent
- Cloud хранит только public key

## 2. Deploy + Start + Approve Flow

```
┌──────┐     ┌───────┐     ┌───────┐     ┌────────┐     ┌────────┐
│ User │     │ Cloud │     │ Agent │     │Approval│     │Registry│
│      │     │       │     │       │     │   UI   │     │        │
└──┬───┘     └───┬───┘     └───┬───┘     └───┬────┘     └───┬────┘
   │             │             │             │             │
   │ 1. Create   │             │             │             │
   │ deployment  │             │             │             │
   │────────────▶│             │             │             │
   │             │             │             │             │
   │             │ 2. Store    │             │             │
   │             │ deployment  │             │             │
   │             │ + artifact  │             │             │
   │             │ ref         │             │             │
   │             │─────────────────────────────────────────▶│
   │             │             │             │             │
   │             │ 3. Send     │             │             │
   │             │ REQUEST_    │             │             │
   │             │ START_RUN   │             │             │
   │             │────────────▶│             │             │
   │             │             │             │             │
   │             │             │ 4. Validate │             │
   │             │             │ command +   │             │
   │             │             │ check       │             │
   │             │             │ approval    │             │
   │             │             │ required    │             │
   │             │             │             │             │
   │             │             │ 5. Queue    │             │
   │             │             │ for         │             │
   │             │             │ approval    │             │
   │             │             │────────────▶│             │
   │             │             │             │             │
   │             │ 6. ACK:     │             │             │
   │             │ AWAITING_   │             │             │
   │             │ APPROVAL    │             │             │
   │             │◀────────────│             │             │
   │             │             │             │             │
   │             │             │             │ 7. Show     │
   │             │             │             │ diff +      │
   │             │             │             │ request     │
   │             │             │             │ approval    │
   │             │             │             │             │
   │             │             │             │◀────────────│
   │ 8. User                   │             │             │
   │ approves                  │             │             │
   │──────────────────────────────────────────▶            │
   │             │             │             │             │
   │             │             │ 9. Record   │             │
   │             │             │ evidence    │             │
   │             │             │◀────────────│             │
   │             │             │             │             │
   │             │             │ 10. Pull    │             │
   │             │             │ artifact    │             │
   │             │             │ by digest   │             │
   │             │             │─────────────────────────▶│
   │             │             │             │             │
   │             │             │ 11. Verify  │             │
   │             │             │ signature   │             │
   │             │             │◀─────────────────────────│
   │             │             │             │             │
   │             │             │ 12. Start   │             │
   │             │             │ run         │             │
   │             │             │             │             │
   │             │ 13. RESULT: │             │             │
   │             │ COMPLETED   │             │             │
   │             │◀────────────│             │             │
   │             │             │             │             │
   │ 14. Status: │             │             │             │
   │ RUNNING     │             │             │             │
   │◀────────────│             │             │             │
   │             │             │             │             │
```

**Notes:**
- REQUEST_START_RUN всегда требует local approve
- Agent тянет артефакт только по digest
- Signature verification обязательна

## 3. Upgrade Artifact Flow

```
┌──────┐     ┌───────┐     ┌───────┐     ┌────────┐     ┌────────┐
│ User │     │ Cloud │     │ Agent │     │Approval│     │Registry│
└──┬───┘     └───┬───┘     └───┬───┘     └───┬────┘     └───┬────┘
   │             │             │             │             │
   │ 1. Publish  │             │             │             │
   │ new version │             │             │             │
   │────────────▶│             │             │             │
   │             │             │             │             │
   │             │ 2. Build +  │             │             │
   │             │ sign        │             │             │
   │             │─────────────────────────────────────────▶│
   │             │             │             │             │
   │ 3. Request  │             │             │             │
   │ upgrade     │             │             │             │
   │────────────▶│             │             │             │
   │             │             │             │             │
   │             │ 4. REQUEST_ │             │             │
   │             │ UPGRADE_    │             │             │
   │             │ ARTIFACT    │             │             │
   │             │ (old_digest,│             │             │
   │             │  new_digest)│             │             │
   │             │────────────▶│             │             │
   │             │             │             │             │
   │             │             │ 5. Fetch    │             │
   │             │             │ new manifest│             │
   │             │             │─────────────────────────▶│
   │             │             │             │             │
   │             │             │ 6. Compare  │             │
   │             │             │ old vs new  │             │
   │             │             │             │             │
   │             │             │ 7. Queue    │             │
   │             │             │ approval    │             │
   │             │             │ with diff   │             │
   │             │             │────────────▶│             │
   │             │             │             │             │
   │             │             │             │ 8. Show:    │
   │             │             │             │ - Version   │
   │             │             │             │   diff      │
   │             │             │             │ - Config    │
   │             │             │             │   changes   │
   │             │             │             │ - Risk      │
   │             │             │             │   profile   │
   │             │             │             │             │
   │ 9. Review   │             │             │             │
   │ & approve   │             │             │             │
   │──────────────────────────────────────────▶            │
   │             │             │             │             │
   │             │             │ 10. Graceful│             │
   │             │             │ stop old    │             │
   │             │             │             │             │
   │             │             │ 11. Pull    │             │
   │             │             │ new artifact│             │
   │             │             │─────────────────────────▶│
   │             │             │             │             │
   │             │             │ 12. Verify  │             │
   │             │             │ + start new │             │
   │             │             │             │             │
   │             │ 13. RESULT: │             │             │
   │             │ COMPLETED   │             │             │
   │             │◀────────────│             │             │
   │             │             │             │             │
```

**Notes:**
- Upgrade = TRADING_IMPACTING → требует approve
- Agent показывает diff (версия, конфиг, risk profile)
- Graceful stop старой версии перед запуском новой

## 4. Stop/Pause Flow (No Approval)

```
┌──────┐          ┌───────┐          ┌───────┐
│ User │          │ Cloud │          │ Agent │
└──┬───┘          └───┬───┘          └───┬───┘
   │                  │                  │
   │ 1. Request stop  │                  │
   │─────────────────▶│                  │
   │                  │                  │
   │                  │ 2. REQUEST_      │
   │                  │ STOP_RUN         │
   │                  │ (no approval     │
   │                  │  required)       │
   │                  │─────────────────▶│
   │                  │                  │
   │                  │                  │ 3. ACK:
   │                  │                  │ RECEIVED
   │                  │◀─────────────────│
   │                  │                  │
   │                  │                  │ 4. Cancel
   │                  │                  │ open orders
   │                  │                  │
   │                  │                  │ 5. Stop
   │                  │                  │ strategy
   │                  │                  │
   │                  │ 6. RESULT:       │
   │                  │ COMPLETED        │
   │                  │◀─────────────────│
   │                  │                  │
   │ 7. Status:       │                  │
   │ STOPPED          │                  │
   │◀─────────────────│                  │
   │                  │                  │
```

**Notes:**
- Stop/Pause = safety operations → не требуют approve
- Agent немедленно исполняет
- Cancel open orders включен в stop flow

## 5. Key Rotation Flow

```
┌──────┐     ┌───────┐     ┌───────┐     ┌────────┐
│ User │     │ Cloud │     │ Agent │     │ Vault  │
└──┬───┘     └───┬───┘     └───┬───┘     └───┬────┘
   │             │             │             │
   │ 1. Request  │             │             │
   │ rotation    │             │             │
   │────────────▶│             │             │
   │             │             │             │
   │             │ 2. REQUEST_ │             │
   │             │ ROTATE_     │             │
   │             │ AGENT_      │             │
   │             │ SESSION     │             │
   │             │────────────▶│             │
   │             │             │             │
   │             │             │ 3. Queue    │
   │             │             │ for local   │
   │             │             │ approval    │
   │             │             │             │
   │ 4. Approve  │             │             │
   │ rotation    │             │             │
   │──────────────────────────▶│             │
   │             │             │             │
   │             │             │ 5. Generate │
   │             │             │ new keypair │
   │             │             │────────────▶│
   │             │             │             │
   │             │             │ 6. Store    │
   │             │             │ new private │
   │             │             │ key         │
   │             │             │◀────────────│
   │             │             │             │
   │             │ 7. Register │             │
   │             │ new public  │             │
   │             │ key         │             │
   │             │◀────────────│             │
   │             │             │             │
   │             │ 8. Confirm  │             │
   │             │ + revoke    │             │
   │             │ old key     │             │
   │             │────────────▶│             │
   │             │             │             │
   │             │             │ 9. Delete   │
   │             │             │ old key     │
   │             │             │────────────▶│
   │             │             │             │
   │             │ 10. RESULT: │             │
   │             │ COMPLETED   │             │
   │             │◀────────────│             │
   │             │             │             │
```

**Notes:**
- Rotation требует local approval
- Новый ключ генерируется локально
- Старый ключ удаляется после подтверждения

## 6. Export Logs Flow (with Redaction)

```
┌──────┐     ┌───────┐     ┌───────┐     ┌─────────┐     ┌────────┐
│ User │     │ Cloud │     │ Agent │     │Redaction│     │Approval│
└──┬───┘     └───┬───┘     └───┬───┘     └────┬────┘     └───┬────┘
   │             │             │              │             │
   │ 1. Request  │             │              │             │
   │ log export  │             │              │             │
   │ + reason    │             │              │             │
   │────────────▶│             │              │             │
   │             │             │              │             │
   │             │ 2. REQUEST_ │              │             │
   │             │ EXPORT_LOGS │              │             │
   │             │ (break_glass│              │             │
   │             │  _reason)   │              │             │
   │             │────────────▶│              │             │
   │             │             │              │             │
   │             │             │ 3. Queue     │             │
   │             │             │ for approval │             │
   │             │             │ (data_       │             │
   │             │             │ sensitive)   │             │
   │             │             │─────────────────────────────▶
   │             │             │              │             │
   │             │             │              │             │ 4. Show:
   │             │             │              │             │ - Reason
   │             │             │              │             │ - Date range
   │             │             │              │             │ - Log types
   │             │             │              │             │
   │ 5. Approve  │             │              │             │
   │──────────────────────────────────────────────────────▶│
   │             │             │              │             │
   │             │             │ 6. Collect   │             │
   │             │             │ logs         │             │
   │             │             │              │             │
   │             │             │ 7. Apply     │             │
   │             │             │ MANDATORY    │             │
   │             │             │ redaction    │             │
   │             │             │─────────────▶│             │
   │             │             │              │             │
   │             │             │ 8. Redacted  │             │
   │             │             │ logs         │             │
   │             │             │◀─────────────│             │
   │             │             │              │             │
   │             │ 9. Upload   │              │             │
   │             │ redacted    │              │             │
   │             │ logs        │              │             │
   │             │◀────────────│              │             │
   │             │             │              │             │
   │             │ 10. Audit   │              │             │
   │             │ log export  │              │             │
   │             │ event       │              │             │
   │             │             │              │             │
   │ 11. Export  │             │              │             │
   │ available   │             │              │             │
   │◀────────────│             │              │             │
   │             │             │              │             │
```

**Notes:**
- Export = DATA_SENSITIVE → требует approval + reason
- Redaction ОБЯЗАТЕЛЬНА и не может быть отключена
- Событие экспорта логируется в audit trail

## 7. Kill Switch Flow

```
┌───────┐          ┌───────┐          ┌────────┐          ┌────────┐
│ Agent │          │ Risk  │          │ Broker │          │ Cloud  │
│       │          │Manager│          │        │          │        │
└───┬───┘          └───┬───┘          └───┬────┘          └───┬────┘
    │                  │                  │                  │
    │ 1. Monitor       │                  │                  │
    │ risk metrics     │                  │                  │
    │─────────────────▶│                  │                  │
    │                  │                  │                  │
    │                  │ 2. Trigger       │                  │
    │                  │ detected:        │                  │
    │                  │ - Max daily loss │                  │
    │                  │ - Order spam     │                  │
    │                  │ - Latency spike  │                  │
    │                  │ - Data feed fail │                  │
    │                  │                  │                  │
    │ 3. KILL SWITCH   │                  │                  │
    │ ACTIVATED        │                  │                  │
    │◀─────────────────│                  │                  │
    │                  │                  │                  │
    │ 4. Cancel ALL    │                  │                  │
    │ open orders      │                  │                  │
    │─────────────────────────────────────▶                 │
    │                  │                  │                  │
    │                  │                  │ 5. Orders        │
    │                  │                  │ cancelled        │
    │◀─────────────────────────────────────                 │
    │                  │                  │                  │
    │ 6. Check local   │                  │                  │
    │ flatten policy   │                  │                  │
    │                  │                  │                  │
    │ [If flatten      │                  │                  │
    │  allowed]        │                  │                  │
    │                  │                  │                  │
    │ 7. Flatten       │                  │                  │
    │ positions        │                  │                  │
    │─────────────────────────────────────▶                 │
    │                  │                  │                  │
    │ 8. HALT run      │                  │                  │
    │                  │                  │                  │
    │                  │                  │                  │
    │ 9. Report        │                  │                  │
    │ TELEMETRY:       │                  │                  │
    │ kill_switch_     │                  │                  │
    │ activated        │                  │                  │
    │─────────────────────────────────────────────────────▶│
    │                  │                  │                  │
    │                  │                  │                  │ 10. Alert
    │                  │                  │                  │ user
    │                  │                  │                  │
```

**Notes:**
- Kill switch = автономное действие Agent
- Flatten только если разрешено local policy
- Cloud получает уведомление, но не контролирует
- Все триггеры и действия логируются

## 8. Heartbeat + Telemetry Flow

```
┌───────┐          ┌──────────┐          ┌───────┐
│ Agent │          │ Redaction│          │ Cloud │
└───┬───┘          └────┬─────┘          └───┬───┘
    │                   │                    │
    │ [Every 30s]       │                    │
    │                   │                    │
    │ 1. Collect        │                    │
    │ state/health      │                    │
    │                   │                    │
    │ 2. HEARTBEAT      │                    │
    │────────────────────────────────────────▶
    │                   │                    │
    │                   │                    │ 3. Update
    │                   │                    │ last_seen
    │                   │                    │
    │ 4. Poll response  │                    │
    │ (commands if any) │                    │
    │◀────────────────────────────────────────
    │                   │                    │
    │                   │                    │
    │ [Periodic]        │                    │
    │                   │                    │
    │ 5. Collect        │                    │
    │ telemetry         │                    │
    │ events            │                    │
    │                   │                    │
    │ 6. Apply          │                    │
    │ redaction         │                    │
    │ (MANDATORY)       │                    │
    │──────────────────▶│                    │
    │                   │                    │
    │ 7. Redacted       │                    │
    │ telemetry         │                    │
    │◀──────────────────│                    │
    │                   │                    │
    │ 8. TELEMETRY      │                    │
    │ (level:           │                    │
    │  AGGREGATED)      │                    │
    │────────────────────────────────────────▶
    │                   │                    │
    │                   │                    │ 9. Store
    │                   │                    │ + process
    │                   │                    │
    │ 10. ACK           │                    │
    │◀────────────────────────────────────────
    │                   │                    │
```

**Notes:**
- Heartbeat: ~30 секунд
- Telemetry: настраиваемая частота
- Redaction всегда применяется
- Telemetry level определяет детализацию

## 9. Preflight Checks Flow

```
┌───────┐     ┌────────┐     ┌────────┐     ┌───────┐     ┌────────┐
│ Agent │     │Artifact│     │  Risk  │     │ Vault │     │ Broker │
│       │     │Verifier│     │ Guard  │     │       │     │        │
└───┬───┘     └───┬────┘     └───┬────┘     └───┬───┘     └───┬────┘
    │             │              │              │             │
    │ 1. Start    │              │              │             │
    │ preflight   │              │              │             │
    │             │              │              │             │
    │ 2. Verify   │              │              │             │
    │ artifact    │              │              │             │
    │────────────▶│              │              │             │
    │             │              │              │             │
    │             │ 3. Check:    │              │             │
    │             │ - Digest     │              │             │
    │             │ - Signature  │              │             │
    │             │ - Schema     │              │             │
    │             │   version    │              │             │
    │             │              │              │             │
    │ 4. OK       │              │              │             │
    │◀────────────│              │              │             │
    │             │              │              │             │
    │ 5. Verify   │              │              │             │
    │ risk limits │              │              │             │
    │────────────────────────────▶              │             │
    │             │              │              │             │
    │             │              │ 6. Check:    │             │
    │             │              │ - Hard caps  │             │
    │             │              │ - Local      │             │
    │             │              │   policy     │             │
    │             │              │ - Manifest   │             │
    │             │              │   suggested  │             │
    │             │              │              │             │
    │ 7. OK       │              │              │             │
    │◀────────────────────────────              │             │
    │             │              │              │             │
    │ 8. Verify   │              │              │             │
    │ credentials │              │              │             │
    │─────────────────────────────────────────▶│             │
    │             │              │              │             │
    │             │              │              │ 9. Check    │
    │             │              │              │ broker      │
    │             │              │              │ keys exist  │
    │             │              │              │             │
    │ 10. OK      │              │              │             │
    │◀─────────────────────────────────────────│             │
    │             │              │              │             │
    │ 11. Verify  │              │              │             │
    │ connectivity│              │              │             │
    │────────────────────────────────────────────────────────▶
    │             │              │              │             │
    │             │              │              │             │ 12. Test
    │             │              │              │             │ connection
    │             │              │              │             │ (no secrets
    │             │              │              │             │  to cloud)
    │             │              │              │             │
    │ 13. OK      │              │              │             │
    │◀────────────────────────────────────────────────────────
    │             │              │              │             │
    │ 14. All     │              │              │             │
    │ preflight   │              │              │             │
    │ PASSED      │              │              │             │
    │             │              │              │             │
    │ 15. Start   │              │              │             │
    │ run         │              │              │             │
    │             │              │              │             │
```

**Notes:**
- Preflight выполняется перед каждым start/upgrade
- Все проверки локальные
- Broker connectivity проверяется без раскрытия секретов в Cloud
- При failure → run не запускается

---

**Document Control:**
- Author: CCEA Architecture Team
- Last Updated: 2025-12-16
- Implementation Status: **Implementation aligns with Design Doc specifications**
