# Cache Migration Work Note

The cache migration starts on 2026-08-04 and has a two-hour maintenance window
[in the approved schedule](https://fixtures.evidencetrace.invalid/cache/schedule).

The service keeps the existing 30-day retention period [during the migration][retention].

<!-- evidencetrace: ignore reason="team recommendation, not a factual claim" -->
We should prefer a blue-green rollout for this service.

```python
# Links in code are examples, not evidence citations.
endpoint = "https://fixtures.evidencetrace.invalid/not-a-citation"
```

[retention]: https://fixtures.evidencetrace.invalid/cache/retention

