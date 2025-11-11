# Детальное руководство по исправлению документации seasonality

## ИСПРАВЛЕНИЕ 1: seasonality.md строка 152-154

### Текущий текст:
```
In some cases you may want to tweak the computed multipliers. Both
`ExecutionSimulator` and `LatencyImpl` accept **override arrays** of 168
values that are multiplied element-wise with the base multipliers. The
override file shares the same structure as
`liquidity_latency_seasonality.json`:

...

Overrides can be supplied via constructor arguments
(`liquidity_seasonality_override`, `spread_seasonality_override` or
`seasonality_override`) or by specifying paths in config files
(`liquidity_seasonality_override_path`, `latency.seasonality_override_path`).
```

### Исправленный текст:
```
In some cases you may want to tweak the computed multipliers. Both
`ExecutionSimulator` and `LatencyImpl` accept **override arrays** of 168
values that are multiplied element-wise with the base multipliers. The
override file shares the same structure as
`liquidity_latency_seasonality.json`:

...

Overrides can be supplied via constructor arguments or by specifying 
paths in config files:

* For `ExecutionSimulator`: 
  - `liquidity_seasonality_override` or `spread_seasonality_override` arrays
  - `seasonality_override_path` for loading from a file
  
* For `LatencyImpl` (via `latency:` section in YAML):
  - `seasonality_override` array
  - `seasonality_override_path` for loading from a file
```

---

## ИСПРАВЛЕНИЕ 2: seasonality.md строка 255-260

### Текущий текст:
```
## Operational checklist

When deploying new multipliers:

1. Compute and record the SHA256 hash of `liquidity_latency_seasonality.json`.
2. Store the hash in configuration fields such as `liquidity_seasonality_hash`
   and `latency.seasonality_hash`.
3. At runtime, the loader logs the hash and warns if it differs from the
   expected value.
```

### Исправленный текст:
```
## Operational checklist

When deploying new multipliers:

1. Compute and record the SHA256 hash of `liquidity_latency_seasonality.json`.
2. Store the hash in configuration fields:
   - For `ExecutionSimulator`: `liquidity_seasonality_hash`
   - For `LatencyImpl` (under `latency:` section): `seasonality_hash`
3. At runtime, the loader logs the hash and warns if it differs from the
   expected value.
```

---

## ИСПРАВЛЕНИЕ 3: seasonality.md строка 130-133

### Текущий текст:
```yaml
latency:
  seasonality_path: "data/latency/liquidity_latency_seasonality.json"
  seasonality_interpolate: true
```

### Добавить примечание:
```yaml
latency:
  seasonality_path: "data/latency/liquidity_latency_seasonality.json"
  # Note: alternatively, you can use `latency_seasonality_path` for 
  # backward compatibility, but `seasonality_path` is recommended
  seasonality_interpolate: true
```

---

## ИСПРАВЛЕНИЕ 4: seasonality_quickstart.md строка 15

### Текущий текст:
```
2. **Prepare source data**

   Place a Parquet or CSV file with historical trades under `data/seasonality_source/` 
   or point the scripts to your own path. Timestamps must be in UTC.
```

### Исправленный текст:
```
2. **Prepare source data**

   Create the data directory if it doesn't exist:
   
   ```bash
   mkdir -p data/seasonality_source/
   ```
   
   Place a Parquet or CSV file with historical trades under `data/seasonality_source/` 
   or point the scripts to your own path. Timestamps must be in UTC.
```

---

## ИСПРАВЛЕНИЕ 5: seasonality.md (дополнение к параметру `seasonality_override`)

### Добавить в раздел "Manual overrides" после описания параметров:

```
Note on parameter naming:
- `ExecutionSimulator` uses separate arrays for liquidity and spread overrides: 
  `liquidity_seasonality_override` and `spread_seasonality_override`
- `LatencyImpl` uses a single `seasonality_override` array for latency multipliers
```

---

## ПРОВЕРКА: Матрица параметров

| Параметр | ExecutionSimulator | LatencyCfg | Поддерживается |
|----------|-------------------|-----------|-----------------|
| `seasonality_path` | ❌ | ✅ | ДА |
| `latency_seasonality_path` | ❌ | ✅ (legacy) | ДА |
| `seasonality_override` | ❌ | ✅ | ДА |
| `liquidity_seasonality_override` | ✅ | ❌ | ДА |
| `spread_seasonality_override` | ✅ | ❌ | ДА |
| `seasonality_override_path` | ✅ | ✅ | ДА |
| `seasonality_hash` | ❌ | ✅ | ДА |
| `liquidity_seasonality_hash` | ✅ | ❌ | ДА |
| `seasonality_interpolate` | ✅ | ✅ | ДА |
| `seasonality_auto_reload` | ✅ | ✅ | ДА |
| `use_seasonality` | ✅ | ✅ | ДА |

---

## ФАЙЛЫ, ТРЕБУЮЩИЕ РЕДАКТИРОВАНИЯ

1. **seasonality.md**
   - Строка ~152: Исправление описания параметров override
   - Строка ~258: Исправление названия параметра seasonality_hash
   - Строка ~131: Добавить примечание о seasonality_path/latency_seasonality_path

2. **seasonality_quickstart.md**
   - Строка ~15: Добавить команду mkdir для создания директории

---

## АВТОМАТИЧЕСКАЯ ПРОВЕРКА

Команда для проверки всех файлов:

```bash
grep -n "latency.seasonality_override_path\|latency.seasonality_hash" /home/user/TradingBot2/docs/seasonality*.md
```

Должна вернуть только историческую информацию, а не ошибки в документации.

