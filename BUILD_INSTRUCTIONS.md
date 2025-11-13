# Инструкции по компиляции Cython модулей

## Требования

Для компиляции модулей необходимы следующие зависимости:

```bash
pip install cython numpy setuptools
```

## Компиляция всех модулей

Для компиляции всех Cython/C++ расширений выполните:

```bash
python setup.py build_ext --inplace
```

Эта команда:
- Компилирует все .pyx файлы в C/C++ код
- Собирает расширения как .so файлы (Linux) или .pyd (Windows)
- Размещает скомпилированные модули в текущей директории (`--inplace`)

## Компиляция конкретного модуля

Если вы изменили только `obs_builder.pyx` и хотите перекомпилировать только его:

```bash
cython obs_builder.pyx
gcc -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing \
    -I/usr/include/python3.12 -o obs_builder.cpython-312-x86_64-linux-gnu.so \
    obs_builder.c
```

Замените `/usr/include/python3.12` на путь к вашим Python заголовочным файлам.

## Проверка компиляции

После компиляции проверьте, что модуль импортируется:

```bash
python3 -c "import obs_builder; print('✓ obs_builder imported successfully')"
```

## Список скомпилированных модулей

Текущие Cython/C++ модули в проекте:

- `obs_builder` - построение observation vector
- `coreworkspace` - workspace для исполнения
- `execevents` - события исполнения
- `execlob_book` - limit order book для исполнения
- `fast_lob` - быстрая реализация LOB (C++)
- `fast_market` - быстрый рыночный симулятор
- `reward` - расчет награды
- `micro_sim` - микроструктурный симулятор
- `marketmarket_simulator_wrapper` - обертка симулятора рынка
- `lob_state_cython` - состояние LOB (Cython)
- `environment` - RL окружение
- `info_builder` - построение info dict
- `risk_manager` - управление рисками

## Troubleshooting

### Ошибка: "Cython is required"

Установите Cython:
```bash
pip install cython
```

### Ошибка: "numpy/arrayobject.h: No such file or directory"

Установите numpy headers:
```bash
pip install numpy
```

### Ошибка компиляции C++

Убедитесь, что у вас установлен компилятор C++17:
```bash
# Linux
sudo apt-get install build-essential

# macOS
xcode-select --install
```

## После изменения obs_builder.pyx

После внесения изменений в `obs_builder.pyx` (в частности, исправлений NaN обработки):

1. Перекомпилируйте модуль:
   ```bash
   python setup.py build_ext --inplace
   ```

2. Запустите тесты:
   ```bash
   python test_nan_reproduction.py
   python test_obs_builder_validation.py
   ```

3. Если тесты проходят, закоммитьте изменения:
   ```bash
   git add obs_builder.pyx obs_builder.c
   git commit -m "fix: complete NaN handling in obs_builder for all indicators"
   git push
   ```

## CI/CD

В CI/CD pipeline компиляция модулей должна выполняться автоматически при изменении .pyx файлов.
