# Sample Predictions

Example outputs from `predict.py` on held-out validation images.

Score: **0 = real photo**, **1 = screen recapture**. Threshold = 0.5.


| File | True Label | Score | Predicted | Result | Latency |
|------|-----------|-------|-----------|--------|---------|
| `real_WhatsApp Image 2026-06-30 at 16.44.11.jpeg` | REAL | 0.0062 | REAL | PASS | 895.1 ms |
| `real_WhatsApp Image 2026-06-30 at 16.44.12 (1).jpeg` | REAL | 0.0216 | REAL | PASS | 113.5 ms |
| `real_WhatsApp Image 2026-06-30 at 16.44.12 (2).jpeg` | REAL | 0.0005 | REAL | PASS | 97.9 ms |
| `screen_WhatsApp Image 2026-06-30 at 16.42.33.jpeg` | SCREEN | 0.9404 | SCREEN | PASS | 109.4 ms |
| `screen_WhatsApp Image 2026-06-30 at 16.42.34 (1).jpeg` | SCREEN | 0.2534 | REAL | FAIL | 98.4 ms |
| `screen_WhatsApp Image 2026-06-30 at 16.42.34 (2).jpeg` | SCREEN | 0.9937 | SCREEN | PASS | 97.2 ms |

## Notes
- Model: MobileNetV2 (frozen, ImageNet weights) + Logistic Regression
- Validation accuracy: **95%** (19/20 correct on held-out val set)
- Latency measured on laptop CPU (inference only, model pre-loaded)
- Score > 0.5 flagged as screen recapture

## Misclassified case
The one FAIL above (score 0.2534) is a high-quality, straight-on screen photo with
minimal moire or glare — the hardest case for any screen-recapture detector.
With more training examples of this type, the model would handle it better.