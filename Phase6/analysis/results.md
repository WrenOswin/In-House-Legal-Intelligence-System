# Phase 6 results (val, 300 held-out train docs, default decoding)

## Table 1 - quality and cost

| run | method | params M | trainable M | ROUGE-1 | ROUGE-2 | ROUGE-L | words | train min | tok/s | VRAM GB | val gen s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_A_lorma | lorma | 144.0 | 3.83 | 0.4723 | 0.2379 | 0.2526 | 382 | 44.9 | 14784 | 4.27 | 246 |
| S2_D_lora | lora | 55.7 | 1.40 | 0.3881 | 0.1474 | 0.2151 | 358 | 34.2 | 19417 | 3.15 | 274 |
| S3_C_houlsby | houlsby adapters | 60.4 | 2.31 | 0.4312 | 0.1931 | 0.2302 | 397 | 52.9 | 12540 | 4.06 | 214 |
| S4_B_fid_pfeiffer | pfeiffer adapters | 74.6 | 1.58 | 0.3740 | 0.1229 | 0.2100 | 430 | 19.7 | 33710 | 1.51 | 137 |
| S5_KD | full fine-tuning | 58.1 | 58.07 | 0.4261 | 0.2032 | 0.2368 | 350 | 55.6 | 11867 | 4.89 | 195 |
| S6_A_full_noKD | full fine-tuning | 58.1 | 58.07 | 0.4585 | 0.2236 | 0.2460 | 385 | 55.2 | 12017 | 4.89 | 199 |
| S7_D_lora_head | lora | 55.7 | 1.40 | 0.3874 | 0.1435 | 0.2146 | 365 | 34.6 | 19189 | 3.15 | 261 |

## Table 2 - coverage, abstractiveness, factuality proxy, redundancy

| run | coverage | novel bigrams | num. halluc. | rep4 | repeated sent. | R1 short ref | R1 mid | R1 long |
|---|---|---|---|---|---|---|---|---|
| S1_A_lorma | 0.273 | 0.329 | 0.305 | 0.0004 | 0.000 | 0.517 | 0.511 | 0.389 |
| S2_D_lora | 0.142 | 0.529 | 0.697 | 0.0015 | 0.000 | 0.410 | 0.431 | 0.323 |
| S3_C_houlsby | 0.211 | 0.422 | 0.585 | 0.0017 | 0.000 | 0.455 | 0.476 | 0.362 |
| S4_B_fid_pfeiffer | 0.125 | 0.580 | 0.788 | 0.0017 | 0.000 | 0.370 | 0.422 | 0.330 |
| S5_KD | 0.225 | 0.401 | 0.527 | 0.0005 | 0.000 | 0.464 | 0.471 | 0.343 |
| S6_A_full_noKD | 0.255 | 0.365 | 0.475 | 0.0007 | 0.000 | 0.493 | 0.505 | 0.378 |
| S7_D_lora_head | 0.141 | 0.536 | 0.688 | 0.0016 | 0.000 | 0.407 | 0.430 | 0.325 |
| reference summaries | 1.000 | 0.271 | 0.176 | 0.0381 | 0.001 | 1.000 | 1.000 | 1.000 |

coverage = share of the reference's content words found in the output; num. halluc. = share of numbers (>= 2 digits) in the output that never occur in the document; R1 by reference-length tercile.
