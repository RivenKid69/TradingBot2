# -*- coding: utf-8 -*-
"""
loaders/ — вендорные free-обогатители панели (Stage D1+).

Каждый модуль (``crypto_enrich``/``equity_enrich``/…) поставляет ``Enricher``-плагины
(см. ``service_xs_data``), которые добавляют реальные колонки (funding/fundamentals/rates/IV)
к собранной панели, чтобы BYO-сигналы «оживали» на бесплатных данных. Все обогатители
PIT-безопасны (as-of join с publish-lag) или честно помечены ``pit_quality``.
"""
