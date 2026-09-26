# Phase 6 Temporal Encoder Variant Report

Epoch 50 is the predeclared principal snapshot. All 66 task/walk/configuration entries replay before this report is written. No universal architecture ranking or model selection is performed.

## Contract

- Substitutions are compared with `H0` at the same 445-dimensional width.
- Additions are compared with both `H0` and the matching 573-dimensional duplicate-CNN control.
- CKA is descriptive only and does not select a configuration.
- Results are single-seed characterisation evidence and remain separated by task, SSL family, and walk.

## Artifacts

Detailed epoch-50 values, paired differences, and resource measurements are stored in the adjacent CSV files. The duplicate controls increase width with perfectly collinear copied coordinates, so they are informative but imperfect capacity controls.

## classification_h2

### Walk 1

| Configuration | Width | macro_f1 | balanced_accuracy |
|---|---:|---:|---:|
| H0 | 445 | 0.45346888 | 0.46578667 |
| HC-SL | 445 | 0.4526284 | 0.46759949 |
| HC-ST | 445 | 0.45295402 | 0.46839649 |
| HB-SL | 445 | 0.45041162 | 0.46588868 |
| HB-ST | 445 | 0.44957447 | 0.46720749 |
| HC-AL | 573 | 0.4495581 | 0.46281062 |
| HC-AT | 573 | 0.44793968 | 0.46105118 |
| HB-AL | 573 | 0.44726599 | 0.46041156 |
| HB-AT | 573 | 0.45013517 | 0.46357014 |
| HC-DC | 573 | 0.44943666 | 0.46238937 |
| HB-DC | 573 | 0.45105379 | 0.4636035 |

Paired differences are candidate minus reference; use the `higher_is_better` field in the CSV when interpreting their direction.

| Comparison | Metric | Delta |
|---|---|---:|
| HC-SL - H0 | macro_f1 | -0.00084047546 |
| HC-SL - H0 | balanced_accuracy | 0.0018128275 |
| HC-ST - H0 | macro_f1 | -0.00051486256 |
| HC-ST - H0 | balanced_accuracy | 0.0026098185 |
| HB-SL - H0 | macro_f1 | -0.0030572629 |
| HB-SL - H0 | balanced_accuracy | 0.00010201265 |
| HB-ST - H0 | macro_f1 | -0.0038944103 |
| HB-ST - H0 | balanced_accuracy | 0.0014208202 |
| HC-AL - H0 | macro_f1 | -0.0039107796 |
| HC-AL - H0 | balanced_accuracy | -0.0029760483 |
| HC-AT - H0 | macro_f1 | -0.0055291966 |
| HC-AT - H0 | balanced_accuracy | -0.0047354883 |
| HC-AL - HC-DC | macro_f1 | 0.00012144361 |
| HC-AL - HC-DC | balanced_accuracy | 0.00042125173 |
| HC-AT - HC-DC | macro_f1 | -0.0014969734 |
| HC-AT - HC-DC | balanced_accuracy | -0.0013381883 |
| HB-AL - H0 | macro_f1 | -0.0062028878 |
| HB-AL - H0 | balanced_accuracy | -0.0053751018 |
| HB-AT - H0 | macro_f1 | -0.0033337126 |
| HB-AT - H0 | balanced_accuracy | -0.0022165311 |
| HB-AL - HB-DC | macro_f1 | -0.0037877933 |
| HB-AL - HB-DC | balanced_accuracy | -0.0031919337 |
| HB-AT - HB-DC | macro_f1 | -0.00091861803 |
| HB-AT - HB-DC | balanced_accuracy | -3.336289e-05 |
### Walk 2

| Configuration | Width | macro_f1 | balanced_accuracy |
|---|---:|---:|---:|
| H0 | 445 | 0.45320192 | 0.4891623 |
| HC-SL | 445 | 0.45583953 | 0.49190193 |
| HC-ST | 445 | 0.45742552 | 0.49538808 |
| HB-SL | 445 | 0.45193394 | 0.48945011 |
| HB-ST | 445 | 0.45869656 | 0.49472524 |
| HC-AL | 573 | 0.45160823 | 0.48102852 |
| HC-AT | 573 | 0.4558025 | 0.48855793 |
| HB-AL | 573 | 0.45423769 | 0.4888595 |
| HB-AT | 573 | 0.45695602 | 0.49026676 |
| HC-DC | 573 | 0.45260312 | 0.48394512 |
| HB-DC | 573 | 0.45431768 | 0.49046375 |

Paired differences are candidate minus reference; use the `higher_is_better` field in the CSV when interpreting their direction.

| Comparison | Metric | Delta |
|---|---|---:|
| HC-SL - H0 | macro_f1 | 0.002637613 |
| HC-SL - H0 | balanced_accuracy | 0.0027396388 |
| HC-ST - H0 | macro_f1 | 0.0042236063 |
| HC-ST - H0 | balanced_accuracy | 0.0062257807 |
| HB-SL - H0 | macro_f1 | -0.0012679721 |
| HB-SL - H0 | balanced_accuracy | 0.00028780985 |
| HB-ST - H0 | macro_f1 | 0.0054946472 |
| HB-ST - H0 | balanced_accuracy | 0.0055629403 |
| HC-AL - H0 | macro_f1 | -0.0015936849 |
| HC-AL - H0 | balanced_accuracy | -0.008133778 |
| HC-AT - H0 | macro_f1 | 0.0026005882 |
| HC-AT - H0 | balanced_accuracy | -0.00060436857 |
| HC-AL - HC-DC | macro_f1 | -0.0009948866 |
| HC-AL - HC-DC | balanced_accuracy | -0.0029166014 |
| HC-AT - HC-DC | macro_f1 | 0.0031993864 |
| HC-AT - HC-DC | balanced_accuracy | 0.004612808 |
| HB-AL - H0 | macro_f1 | 0.001035771 |
| HB-AL - H0 | balanced_accuracy | -0.0003027966 |
| HB-AT - H0 | macro_f1 | 0.0037541022 |
| HB-AT - H0 | balanced_accuracy | 0.0011044662 |
| HB-AL - HB-DC | macro_f1 | -7.9988661e-05 |
| HB-AL - HB-DC | balanced_accuracy | -0.0016042487 |
| HB-AT - HB-DC | macro_f1 | 0.0026383426 |
| HB-AT - HB-DC | balanced_accuracy | -0.00019698594 |

## absolute_price_h8

### Walk 1

| Configuration | Width | price_mae | price_rmse | price_pearson | price_spearman | implied_pearson | implied_spearman | implied_sign_agreement | cross_sectional_rank_ic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H0 | 445 | 0.0079853986 | 0.018905834 | 0.99619793 | 0.95107774 | 0.20942128 | 0.14736288 | 0.42619159 | 0.14395593 |
| HC-SL | 445 | 0.0070580732 | 0.01522549 | 0.99759275 | 0.952572 | 0.25494957 | 0.12701096 | 0.57431483 | 0.12254269 |
| HC-ST | 445 | 0.0072008151 | 0.015492862 | 0.99751431 | 0.91800825 | 0.24367404 | 0.11663187 | 0.55393535 | 0.11811053 |
| HB-SL | 445 | 0.0072121884 | 0.015714879 | 0.99734867 | 0.94364731 | 0.22300654 | 0.12893584 | 0.56996662 | 0.13213639 |
| HB-ST | 445 | 0.0077641288 | 0.016987886 | 0.99695238 | 0.94024488 | 0.22410012 | 0.12298302 | 0.57093289 | 0.12184638 |
| HC-AL | 573 | 0.0069333417 | 0.01518399 | 0.99754624 | 0.95082354 | 0.2572802 | 0.13059156 | 0.57927793 | 0.12701169 |
| HC-AT | 573 | 0.007097952 | 0.01540835 | 0.99745403 | 0.93275848 | 0.24767202 | 0.11393073 | 0.55279339 | 0.11710502 |
| HB-AL | 573 | 0.0070331705 | 0.015303672 | 0.99755664 | 0.95579861 | 0.2379846 | 0.1368096 | 0.58446065 | 0.13772723 |
| HB-AT | 573 | 0.0076383568 | 0.016491021 | 0.99717149 | 0.9344546 | 0.2277057 | 0.11634736 | 0.56403725 | 0.11507089 |
| HC-DC | 573 | 0.0081143313 | 0.019393603 | 0.99598474 | 0.95352681 | 0.19978343 | 0.151268 | 0.56438862 | 0.14865659 |
| HB-DC | 573 | 0.0076099973 | 0.018070706 | 0.99647913 | 0.95658141 | 0.21290603 | 0.15653961 | 0.57268974 | 0.15564276 |

Paired differences are candidate minus reference; use the `higher_is_better` field in the CSV when interpreting their direction.

| Comparison | Metric | Delta |
|---|---|---:|
| HC-SL - H0 | price_mae | -0.00092732537 |
| HC-SL - H0 | price_rmse | -0.0036803438 |
| HC-SL - H0 | price_pearson | 0.0013948228 |
| HC-SL - H0 | price_spearman | 0.0014942544 |
| HC-SL - H0 | implied_pearson | 0.045528294 |
| HC-SL - H0 | implied_spearman | -0.020351927 |
| HC-SL - H0 | implied_sign_agreement | 0.14812323 |
| HC-SL - H0 | cross_sectional_rank_ic | -0.02141324 |
| HC-ST - H0 | price_mae | -0.00078458354 |
| HC-ST - H0 | price_rmse | -0.0034129717 |
| HC-ST - H0 | price_pearson | 0.0013163839 |
| HC-ST - H0 | price_spearman | -0.033069496 |
| HC-ST - H0 | implied_pearson | 0.034252767 |
| HC-ST - H0 | implied_spearman | -0.030731018 |
| HC-ST - H0 | implied_sign_agreement | 0.12774375 |
| HC-ST - H0 | cross_sectional_rank_ic | -0.025845401 |
| HB-SL - H0 | price_mae | -0.00077321022 |
| HB-SL - H0 | price_rmse | -0.0031909548 |
| HB-SL - H0 | price_pearson | 0.0011507466 |
| HB-SL - H0 | price_spearman | -0.0074304283 |
| HB-SL - H0 | implied_pearson | 0.013585261 |
| HB-SL - H0 | implied_spearman | -0.018427047 |
| HB-SL - H0 | implied_sign_agreement | 0.14377503 |
| HB-SL - H0 | cross_sectional_rank_ic | -0.011819545 |
| HB-ST - H0 | price_mae | -0.00022126983 |
| HB-ST - H0 | price_rmse | -0.0019179477 |
| HB-ST - H0 | price_pearson | 0.000754455 |
| HB-ST - H0 | price_spearman | -0.010832859 |
| HB-ST - H0 | implied_pearson | 0.014678849 |
| HB-ST - H0 | implied_spearman | -0.024379862 |
| HB-ST - H0 | implied_sign_agreement | 0.14474129 |
| HB-ST - H0 | cross_sectional_rank_ic | -0.022109556 |
| HC-AL - H0 | price_mae | -0.0010520569 |
| HC-AL - H0 | price_rmse | -0.0037218442 |
| HC-AL - H0 | price_pearson | 0.0013483152 |
| HC-AL - H0 | price_spearman | -0.00025419909 |
| HC-AL - H0 | implied_pearson | 0.047858929 |
| HC-AL - H0 | implied_spearman | -0.016771326 |
| HC-AL - H0 | implied_sign_agreement | 0.15308634 |
| HC-AL - H0 | cross_sectional_rank_ic | -0.016944239 |
| HC-AT - H0 | price_mae | -0.0008874466 |
| HC-AT - H0 | price_rmse | -0.0034974837 |
| HC-AT - H0 | price_pearson | 0.0012561087 |
| HC-AT - H0 | price_spearman | -0.018319263 |
| HC-AT - H0 | implied_pearson | 0.038250743 |
| HC-AT - H0 | implied_spearman | -0.033432152 |
| HC-AT - H0 | implied_sign_agreement | 0.1266018 |
| HC-AT - H0 | cross_sectional_rank_ic | -0.026850913 |
| HC-AL - HC-DC | price_mae | -0.0011809896 |
| HC-AL - HC-DC | price_rmse | -0.0042096133 |
| HC-AL - HC-DC | price_pearson | 0.0015615039 |
| HC-AL - HC-DC | price_spearman | -0.0027032648 |
| HC-AL - HC-DC | implied_pearson | 0.05749677 |
| HC-AL - HC-DC | implied_spearman | -0.020676443 |
| HC-AL - HC-DC | implied_sign_agreement | 0.014889318 |
| HC-AL - HC-DC | cross_sectional_rank_ic | -0.021644892 |
| HC-AT - HC-DC | price_mae | -0.0010163793 |
| HC-AT - HC-DC | price_rmse | -0.0039852528 |
| HC-AT - HC-DC | price_pearson | 0.0014692974 |
| HC-AT - HC-DC | price_spearman | -0.020768329 |
| HC-AT - HC-DC | implied_pearson | 0.047888583 |
| HC-AT - HC-DC | implied_spearman | -0.037337269 |
| HC-AT - HC-DC | implied_sign_agreement | -0.011595221 |
| HC-AT - HC-DC | cross_sectional_rank_ic | -0.031551566 |
| HB-AL - H0 | price_mae | -0.00095222807 |
| HB-AL - H0 | price_rmse | -0.003602162 |
| HB-AL - H0 | price_pearson | 0.0013587103 |
| HB-AL - H0 | price_spearman | 0.0047208688 |
| HB-AL - H0 | implied_pearson | 0.02856332 |
| HB-AL - H0 | implied_spearman | -0.010553282 |
| HB-AL - H0 | implied_sign_agreement | 0.15826905 |
| HB-AL - H0 | cross_sectional_rank_ic | -0.0062286999 |
| HB-AT - H0 | price_mae | -0.00034704182 |
| HB-AT - H0 | price_rmse | -0.0024148123 |
| HB-AT - H0 | price_pearson | 0.00097356253 |
| HB-AT - H0 | price_spearman | -0.016623146 |
| HB-AT - H0 | implied_pearson | 0.018284422 |
| HB-AT - H0 | implied_spearman | -0.031015522 |
| HB-AT - H0 | implied_sign_agreement | 0.13784565 |
| HB-AT - H0 | cross_sectional_rank_ic | -0.028885041 |
| HB-AL - HB-DC | price_mae | -0.00057682676 |
| HB-AL - HB-DC | price_rmse | -0.0027670339 |
| HB-AL - HB-DC | price_pearson | 0.0010775054 |
| HB-AL - HB-DC | price_spearman | -0.0007828034 |
| HB-AL - HB-DC | implied_pearson | 0.025078563 |
| HB-AL - HB-DC | implied_spearman | -0.019730012 |
| HB-AL - HB-DC | implied_sign_agreement | 0.011770907 |
| HB-AL - HB-DC | cross_sectional_rank_ic | -0.017915524 |
| HB-AT - HB-DC | price_mae | 2.8359478e-05 |
| HB-AT - HB-DC | price_rmse | -0.0015796842 |
| HB-AT - HB-DC | price_pearson | 0.00069235759 |
| HB-AT - HB-DC | price_spearman | -0.022126818 |
| HB-AT - HB-DC | implied_pearson | 0.014799665 |
| HB-AT - HB-DC | implied_spearman | -0.040192252 |
| HB-AT - HB-DC | implied_sign_agreement | -0.0086524947 |
| HB-AT - HB-DC | cross_sectional_rank_ic | -0.040571865 |
### Walk 2

| Configuration | Width | price_mae | price_rmse | price_pearson | price_spearman | implied_pearson | implied_spearman | implied_sign_agreement | cross_sectional_rank_ic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H0 | 445 | 0.010053535 | 0.030072603 | 0.98754073 | 0.98232066 | -0.0035409501 | 0.099988083 | 0.41122464 | 0.094730116 |
| HC-SL | 445 | 0.0079104359 | 0.025082657 | 0.99127634 | 0.98485487 | -0.012600275 | 0.10858022 | 0.55540014 | 0.10083202 |
| HC-ST | 445 | 0.0088678847 | 0.026922494 | 0.98995094 | 0.97651253 | -0.025976644 | 0.080318583 | 0.55220302 | 0.075045971 |
| HB-SL | 445 | 0.0081996498 | 0.025302849 | 0.99117434 | 0.98173649 | 0.018020136 | 0.10623264 | 0.53601758 | 0.11095226 |
| HB-ST | 445 | 0.0084958078 | 0.026898534 | 0.99028029 | 0.98162257 | -0.0088685724 | 0.094307158 | 0.55769807 | 0.079402092 |
| HC-AL | 573 | 0.0080423219 | 0.024919054 | 0.99139732 | 0.98619119 | 0.014197874 | 0.11481839 | 0.5451094 | 0.10138724 |
| HC-AT | 573 | 0.0087611243 | 0.02684579 | 0.98999572 | 0.98141906 | 0.0033920972 | 0.079968527 | 0.54540913 | 0.067163658 |
| HB-AL | 573 | 0.0078879082 | 0.024590055 | 0.99163141 | 0.98588371 | 0.014756734 | 0.11019901 | 0.54840643 | 0.10674112 |
| HB-AT | 573 | 0.0082991478 | 0.025418239 | 0.99118254 | 0.98170051 | 0.0086285889 | 0.09264589 | 0.54171246 | 0.071207054 |
| HC-DC | 573 | 0.010505443 | 0.031628597 | 0.98620657 | 0.98377036 | 0.0066839539 | 0.098271426 | 0.5540014 | 0.096456426 |
| HB-DC | 573 | 0.010227382 | 0.029588354 | 0.98810341 | 0.98320561 | -0.011845992 | 0.083947461 | 0.55110401 | 0.079052375 |

Paired differences are candidate minus reference; use the `higher_is_better` field in the CSV when interpreting their direction.

| Comparison | Metric | Delta |
|---|---|---:|
| HC-SL - H0 | price_mae | -0.0021430991 |
| HC-SL - H0 | price_rmse | -0.0049899462 |
| HC-SL - H0 | price_pearson | 0.003735612 |
| HC-SL - H0 | price_spearman | 0.0025342165 |
| HC-SL - H0 | implied_pearson | -0.0090593253 |
| HC-SL - H0 | implied_spearman | 0.008592132 |
| HC-SL - H0 | implied_sign_agreement | 0.1441755 |
| HC-SL - H0 | cross_sectional_rank_ic | 0.0061019057 |
| HC-ST - H0 | price_mae | -0.0011856503 |
| HC-ST - H0 | price_rmse | -0.0031501091 |
| HC-ST - H0 | price_pearson | 0.0024102046 |
| HC-ST - H0 | price_spearman | -0.0058081228 |
| HC-ST - H0 | implied_pearson | -0.022435694 |
| HC-ST - H0 | implied_spearman | -0.0196695 |
| HC-ST - H0 | implied_sign_agreement | 0.14097838 |
| HC-ST - H0 | cross_sectional_rank_ic | -0.019684144 |
| HB-SL - H0 | price_mae | -0.0018538852 |
| HB-SL - H0 | price_rmse | -0.0047697536 |
| HB-SL - H0 | price_pearson | 0.0036336098 |
| HB-SL - H0 | price_spearman | -0.00058416689 |
| HB-SL - H0 | implied_pearson | 0.021561086 |
| HB-SL - H0 | implied_spearman | 0.0062445577 |
| HB-SL - H0 | implied_sign_agreement | 0.12479294 |
| HB-SL - H0 | cross_sectional_rank_ic | 0.01622214 |
| HB-ST - H0 | price_mae | -0.0015577271 |
| HB-ST - H0 | price_rmse | -0.0031740685 |
| HB-ST - H0 | price_pearson | 0.0027395613 |
| HB-ST - H0 | price_spearman | -0.00069808555 |
| HB-ST - H0 | implied_pearson | -0.0053276223 |
| HB-ST - H0 | implied_spearman | -0.0056809254 |
| HB-ST - H0 | implied_sign_agreement | 0.14647343 |
| HB-ST - H0 | cross_sectional_rank_ic | -0.015328024 |
| HC-AL - H0 | price_mae | -0.0020112131 |
| HC-AL - H0 | price_rmse | -0.0051535491 |
| HC-AL - H0 | price_pearson | 0.003856585 |
| HC-AL - H0 | price_spearman | 0.0038705311 |
| HC-AL - H0 | implied_pearson | 0.017738824 |
| HC-AL - H0 | implied_spearman | 0.014830304 |
| HC-AL - H0 | implied_sign_agreement | 0.13388476 |
| HC-AL - H0 | cross_sectional_rank_ic | 0.006657123 |
| HC-AT - H0 | price_mae | -0.0012924107 |
| HC-AT - H0 | price_rmse | -0.0032268124 |
| HC-AT - H0 | price_pearson | 0.0024549881 |
| HC-AT - H0 | price_spearman | -0.00090159411 |
| HC-AT - H0 | implied_pearson | 0.0069330473 |
| HC-AT - H0 | implied_spearman | -0.020019556 |
| HC-AT - H0 | implied_sign_agreement | 0.13418449 |
| HC-AT - H0 | cross_sectional_rank_ic | -0.027566458 |
| HC-AL - HC-DC | price_mae | -0.0024631211 |
| HC-AL - HC-DC | price_rmse | -0.0067095431 |
| HC-AL - HC-DC | price_pearson | 0.0051907509 |
| HC-AL - HC-DC | price_spearman | 0.0024208325 |
| HC-AL - HC-DC | implied_pearson | 0.0075139197 |
| HC-AL - HC-DC | implied_spearman | 0.01654696 |
| HC-AL - HC-DC | implied_sign_agreement | -0.0088919972 |
| HC-AL - HC-DC | cross_sectional_rank_ic | 0.0049308127 |
| HC-AT - HC-DC | price_mae | -0.0017443187 |
| HC-AT - HC-DC | price_rmse | -0.0047828064 |
| HC-AT - HC-DC | price_pearson | 0.0037891541 |
| HC-AT - HC-DC | price_spearman | -0.0023512926 |
| HC-AT - HC-DC | implied_pearson | -0.0032918567 |
| HC-AT - HC-DC | implied_spearman | -0.018302899 |
| HC-AT - HC-DC | implied_sign_agreement | -0.008592267 |
| HC-AT - HC-DC | cross_sectional_rank_ic | -0.029292768 |
| HB-AL - H0 | price_mae | -0.0021656268 |
| HB-AL - H0 | price_rmse | -0.0054825475 |
| HB-AL - H0 | price_pearson | 0.004090676 |
| HB-AL - H0 | price_spearman | 0.0035630537 |
| HB-AL - H0 | implied_pearson | 0.018297684 |
| HB-AL - H0 | implied_spearman | 0.01021093 |
| HB-AL - H0 | implied_sign_agreement | 0.13718179 |
| HB-AL - H0 | cross_sectional_rank_ic | 0.012011002 |
| HB-AT - H0 | price_mae | -0.0017543872 |
| HB-AT - H0 | price_rmse | -0.0046543642 |
| HB-AT - H0 | price_pearson | 0.0036418083 |
| HB-AT - H0 | price_spearman | -0.00062015061 |
| HB-AT - H0 | implied_pearson | 0.012169539 |
| HB-AT - H0 | implied_spearman | -0.007342193 |
| HB-AT - H0 | implied_sign_agreement | 0.13048782 |
| HB-AT - H0 | cross_sectional_rank_ic | -0.023523061 |
| HB-AL - HB-DC | price_mae | -0.0023394737 |
| HB-AL - HB-DC | price_rmse | -0.0049982987 |
| HB-AL - HB-DC | price_pearson | 0.0035279958 |
| HB-AL - HB-DC | price_spearman | 0.0026780994 |
| HB-AL - HB-DC | implied_pearson | 0.026602726 |
| HB-AL - HB-DC | implied_spearman | 0.026251552 |
| HB-AL - HB-DC | implied_sign_agreement | -0.0026975722 |
| HB-AL - HB-DC | cross_sectional_rank_ic | 0.027688742 |
| HB-AT - HB-DC | price_mae | -0.0019282341 |
| HB-AT - HB-DC | price_rmse | -0.0041701155 |
| HB-AT - HB-DC | price_pearson | 0.003079128 |
| HB-AT - HB-DC | price_spearman | -0.0015051049 |
| HB-AT - HB-DC | implied_pearson | 0.020474581 |
| HB-AT - HB-DC | implied_spearman | 0.0086984287 |
| HB-AT - HB-DC | implied_sign_agreement | -0.0093915476 |
| HB-AT - HB-DC | cross_sectional_rank_ic | -0.0078453212 |

## realised_variance

### Walk 1

| Configuration | Width | mae | rmse | mse | pearson | spearman |
|---|---:|---:|---:|---:|---:|---:|
| H0 | 445 | 0.0006261896 | 0.025583491 | 0.00065451501 | 0.028055832 | 0.48291385 |
| HC-SL | 445 | 0.00062555105 | 0.025583235 | 0.0006545019 | 0.02971497 | 0.48440544 |
| HC-ST | 445 | 0.00062955527 | 0.025582217 | 0.00065444983 | 0.031697733 | 0.48643398 |
| HB-SL | 445 | 0.00062790377 | 0.025582896 | 0.00065448455 | 0.029348946 | 0.50220058 |
| HB-ST | 445 | 0.0006251861 | 0.025581846 | 0.00065443086 | 0.032752342 | 0.45569874 |
| HC-AL | 573 | 0.00062747113 | 0.02558325 | 0.00065450266 | 0.028894778 | 0.49302257 |
| HC-AT | 573 | 0.0006297071 | 0.02558309 | 0.00065449451 | 0.028873334 | 0.49453385 |
| HB-AL | 573 | 0.00062832784 | 0.025582519 | 0.00065446525 | 0.030302429 | 0.509538 |
| HB-AT | 573 | 0.00062704979 | 0.025583356 | 0.0006545081 | 0.028625273 | 0.48043998 |
| HC-DC | 573 | 0.00062532494 | 0.025583515 | 0.00065451622 | 0.028157369 | 0.48804536 |
| HB-DC | 573 | 0.00062430042 | 0.025583547 | 0.00065451785 | 0.028948249 | 0.47753565 |

Paired differences are candidate minus reference; use the `higher_is_better` field in the CSV when interpreting their direction.

| Comparison | Metric | Delta |
|---|---|---:|
| HC-SL - H0 | mae | -6.3854701e-07 |
| HC-SL - H0 | rmse | -2.5618869e-07 |
| HC-SL - H0 | mse | -1.3108337e-08 |
| HC-SL - H0 | pearson | 0.0016591371 |
| HC-SL - H0 | spearman | 0.0014915906 |
| HC-ST - H0 | mae | 3.3656683e-06 |
| HC-ST - H0 | rmse | -1.2739675e-06 |
| HC-ST - H0 | mse | -6.518345e-08 |
| HC-ST - H0 | pearson | 0.0036419005 |
| HC-ST - H0 | spearman | 0.003520125 |
| HB-SL - H0 | mae | 1.7141702e-06 |
| HB-SL - H0 | rmse | -5.9540246e-07 |
| HB-SL - H0 | mse | -3.0464593e-08 |
| HB-SL - H0 | pearson | 0.0012931131 |
| HB-SL - H0 | spearman | 0.019286726 |
| HB-ST - H0 | mae | -1.0035042e-06 |
| HB-ST - H0 | rmse | -1.6446715e-06 |
| HB-ST - H0 | mse | -8.4150173e-08 |
| HB-ST - H0 | pearson | 0.0046965097 |
| HB-ST - H0 | spearman | -0.02721511 |
| HC-AL - H0 | mae | 1.2815279e-06 |
| HC-AL - H0 | rmse | -2.4150402e-07 |
| HC-AL - H0 | mse | -1.2356974e-08 |
| HC-AL - H0 | pearson | 0.00083894588 |
| HC-AL - H0 | spearman | 0.010108722 |
| HC-AT - H0 | mae | 3.517498e-06 |
| HC-AT - H0 | rmse | -4.0073052e-07 |
| HC-AT - H0 | mse | -2.0504011e-08 |
| HC-AT - H0 | pearson | 0.00081750131 |
| HC-AT - H0 | spearman | 0.01162 |
| HC-AL - HC-DC | mae | 2.1461881e-06 |
| HC-AL - HC-DC | rmse | -2.6504483e-07 |
| HC-AL - HC-DC | mse | -1.3561486e-08 |
| HC-AL - HC-DC | pearson | 0.00073740967 |
| HC-AL - HC-DC | spearman | 0.0049772152 |
| HC-AT - HC-DC | mae | 4.3821581e-06 |
| HC-AT - HC-DC | rmse | -4.2427133e-07 |
| HC-AT - HC-DC | mse | -2.1708523e-08 |
| HC-AT - HC-DC | pearson | 0.0007159651 |
| HC-AT - HC-DC | spearman | 0.0064884935 |
| HB-AL - H0 | mae | 2.1382394e-06 |
| HB-AL - H0 | rmse | -9.7247534e-07 |
| HB-AL - H0 | mse | -4.9757683e-08 |
| HB-AL - H0 | pearson | 0.0022465964 |
| HB-AL - H0 | spearman | 0.026624152 |
| HB-AT - H0 | mae | 8.6018828e-07 |
| HB-AT - H0 | rmse | -1.3506597e-07 |
| HB-AT - H0 | mse | -6.9108997e-09 |
| HB-AT - H0 | pearson | 0.00056944089 |
| HB-AT - H0 | spearman | -0.0024738698 |
| HB-AL - HB-DC | mae | 4.0274185e-06 |
| HB-AL - HB-DC | rmse | -1.0279997e-06 |
| HB-AL - HB-DC | mse | -5.2598701e-08 |
| HB-AL - HB-DC | pearson | 0.0013541802 |
| HB-AL - HB-DC | spearman | 0.032002358 |
| HB-AT - HB-DC | mae | 2.7493673e-06 |
| HB-AT - HB-DC | rmse | -1.9059035e-07 |
| HB-AT - HB-DC | mse | -9.7519177e-09 |
| HB-AT - HB-DC | pearson | -0.00032297532 |
| HB-AT - HB-DC | spearman | 0.0029043367 |
### Walk 2

| Configuration | Width | mae | rmse | mse | pearson | spearman |
|---|---:|---:|---:|---:|---:|---:|
| H0 | 445 | 0.00050204011 | 0.0098587248 | 9.7194455e-05 | 0.0016464696 | 0.57134454 |
| HC-SL | 445 | 0.0005110997 | 0.0098594289 | 9.7208339e-05 | 0.0041268321 | 0.57277873 |
| HC-ST | 445 | 0.00051339842 | 0.0098594582 | 9.7208916e-05 | 0.0050214799 | 0.57781979 |
| HB-SL | 445 | 0.00051087675 | 0.0098613921 | 9.7247055e-05 | 0.00043294917 | 0.57067269 |
| HB-ST | 445 | 0.00051045615 | 0.0098572239 | 9.7164863e-05 | 0.010972492 | 0.57607759 |
| HC-AL | 573 | 0.00050474589 | 0.0098590731 | 9.7201323e-05 | 0.0050592538 | 0.56755362 |
| HC-AT | 573 | 0.00051007465 | 0.0098622268 | 9.7263517e-05 | -0.0003437773 | 0.56225711 |
| HB-AL | 573 | 0.00050536317 | 0.0098603734 | 9.7226964e-05 | 0.00092870407 | 0.56852041 |
| HB-AT | 573 | 0.00051033624 | 0.0098570929 | 9.716228e-05 | 0.01048398 | 0.57485652 |
| HC-DC | 573 | 0.00049932424 | 0.0098582396 | 9.7184888e-05 | 0.0030475864 | 0.57207603 |
| HB-DC | 573 | 0.00050198247 | 0.0098571843 | 9.7164082e-05 | 0.0059639373 | 0.56971896 |

Paired differences are candidate minus reference; use the `higher_is_better` field in the CSV when interpreting their direction.

| Comparison | Metric | Delta |
|---|---|---:|
| HC-SL - H0 | mae | 9.0595885e-06 |
| HC-SL - H0 | rmse | 7.041117e-07 |
| HC-SL - H0 | mse | 1.3883783e-08 |
| HC-SL - H0 | pearson | 0.0024803625 |
| HC-SL - H0 | spearman | 0.0014341939 |
| HC-ST - H0 | mae | 1.1358307e-05 |
| HC-ST - H0 | rmse | 7.3336519e-07 |
| HC-ST - H0 | mse | 1.4460629e-08 |
| HC-ST - H0 | pearson | 0.0033750104 |
| HC-ST - H0 | spearman | 0.0064752511 |
| HB-SL - H0 | mae | 8.8366343e-06 |
| HB-SL - H0 | rmse | 2.6673132e-06 |
| HB-SL - H0 | mse | 5.2599729e-08 |
| HB-SL - H0 | pearson | -0.0012135204 |
| HB-SL - H0 | spearman | -0.00067184959 |
| HB-ST - H0 | mae | 8.4160332e-06 |
| HB-ST - H0 | rmse | -1.5009469e-06 |
| HB-ST - H0 | mse | -2.9592593e-08 |
| HB-ST - H0 | pearson | 0.0093260228 |
| HB-ST - H0 | spearman | 0.0047330532 |
| HC-AL - H0 | mae | 2.7057717e-06 |
| HC-AL - H0 | rmse | 3.4828426e-07 |
| HC-AL - H0 | mse | 6.8673987e-09 |
| HC-AL - H0 | pearson | 0.0034127842 |
| HC-AL - H0 | spearman | -0.0037909198 |
| HC-AT - H0 | mae | 8.0345303e-06 |
| HC-AT - H0 | rmse | 3.5019254e-06 |
| HC-AT - H0 | mse | 6.9061301e-08 |
| HC-AT - H0 | pearson | -0.0019902469 |
| HC-AT - H0 | spearman | -0.0090874265 |
| HC-AL - HC-DC | mae | 5.4216508e-06 |
| HC-AL - HC-DC | rmse | 8.3352987e-07 |
| HC-AL - HC-DC | mse | 1.6434969e-08 |
| HC-AL - HC-DC | pearson | 0.0020116674 |
| HC-AL - HC-DC | spearman | -0.0045224068 |
| HC-AT - HC-DC | mae | 1.0750409e-05 |
| HC-AT - HC-DC | rmse | 3.987171e-06 |
| HC-AT - HC-DC | mse | 7.8628872e-08 |
| HC-AT - HC-DC | pearson | -0.0033913637 |
| HC-AT - HC-DC | spearman | -0.0098189135 |
| HB-AL - H0 | mae | 3.3230522e-06 |
| HB-AL - H0 | rmse | 1.6486195e-06 |
| HB-AL - H0 | mse | 3.250929e-08 |
| HB-AL - H0 | pearson | -0.0007177655 |
| HB-AL - H0 | spearman | -0.0028241322 |
| HB-AT - H0 | mae | 8.2961205e-06 |
| HB-AT - H0 | rmse | -1.6319433e-06 |
| HB-AT - H0 | mse | -3.2175096e-08 |
| HB-AT - H0 | pearson | 0.0088375105 |
| HB-AT - H0 | spearman | 0.0035119817 |
| HB-AL - HB-DC | mae | 3.3806983e-06 |
| HB-AL - HB-DC | rmse | 3.1891781e-06 |
| HB-AL - HB-DC | mse | 6.2882803e-08 |
| HB-AL - HB-DC | pearson | -0.0050352332 |
| HB-AL - HB-DC | spearman | -0.0011985498 |
| HB-AT - HB-DC | mae | 8.3537666e-06 |
| HB-AT - HB-DC | rmse | -9.1384654e-08 |
| HB-AT - HB-DC | mse | -1.8015824e-09 |
| HB-AT - HB-DC | pearson | 0.0045200428 |
| HB-AT - HB-DC | spearman | 0.0051375641 |

## Non-learned task references

| Task | Walk | Reference | Metric | Value |
|---|---:|---|---|---:|
| classification_h2 | 1 | always_stable_reference | macro_f1 | 0.28277154 |
| classification_h2 | 1 | always_stable_reference | balanced_accuracy | 0.33333333 |
| classification_h2 | 1 | repeated_training_prior_reference | macro_f1 | 0.28277154 |
| classification_h2 | 1 | repeated_training_prior_reference | balanced_accuracy | 0.33333333 |
| classification_h2 | 2 | always_stable_reference | macro_f1 | 0.28388324 |
| classification_h2 | 2 | always_stable_reference | balanced_accuracy | 0.33333333 |
| classification_h2 | 2 | repeated_training_prior_reference | macro_f1 | 0.28388324 |
| classification_h2 | 2 | repeated_training_prior_reference | balanced_accuracy | 0.33333333 |
| absolute_price_h8 | 1 | current_price_persistence | price_mae | 0.0029782621 |
| absolute_price_h8 | 1 | current_price_persistence | price_rmse | 0.011871641 |
| absolute_price_h8 | 1 | current_price_persistence | price_pearson | 0.99835337 |
| absolute_price_h8 | 1 | current_price_persistence | price_spearman | 0.98060084 |
| absolute_price_h8 | 1 | last_hour_reversal | cross_sectional_rank_ic | 0.30123754 |
| absolute_price_h8 | 2 | current_price_persistence | price_mae | 0.004426176 |
| absolute_price_h8 | 2 | current_price_persistence | price_rmse | 0.020395383 |
| absolute_price_h8 | 2 | current_price_persistence | price_pearson | 0.99424008 |
| absolute_price_h8 | 2 | current_price_persistence | price_spearman | 0.99230436 |
| absolute_price_h8 | 2 | last_hour_reversal | cross_sectional_rank_ic | 0.24001068 |
| realised_variance | 1 | historical_persistence | mae | 0.0010663084 |
| realised_variance | 1 | historical_persistence | rmse | 0.035566885 |
| realised_variance | 1 | historical_persistence | mse | 0.0012650033 |
| realised_variance | 1 | historical_persistence | pearson | 0.033861437 |
| realised_variance | 1 | historical_persistence | spearman | 0.55861871 |
| realised_variance | 1 | training_median | mae | 0.00066994841 |
| realised_variance | 1 | training_median | rmse | 0.025594413 |
| realised_variance | 1 | training_median | mse | 0.00065507399 |
| realised_variance | 1 | training_median | pearson | nan |
| realised_variance | 1 | training_median | spearman | nan |
| realised_variance | 1 | zero | mae | 0.00066157896 |
| realised_variance | 1 | zero | rmse | 0.025595147 |
| realised_variance | 1 | zero | mse | 0.00065511153 |
| realised_variance | 1 | zero | pearson | nan |
| realised_variance | 1 | zero | spearman | nan |
| realised_variance | 2 | historical_persistence | mae | 0.00056671967 |
| realised_variance | 2 | historical_persistence | rmse | 0.010025514 |
| realised_variance | 2 | historical_persistence | mse | 0.00010051094 |
| realised_variance | 2 | historical_persistence | pearson | 0.0051886723 |
| realised_variance | 2 | historical_persistence | spearman | 0.61675031 |
| realised_variance | 2 | training_median | mae | 0.0005113091 |
| realised_variance | 2 | training_median | rmse | 0.0098596014 |
| realised_variance | 2 | training_median | mse | 9.721174e-05 |
| realised_variance | 2 | training_median | pearson | nan |
| realised_variance | 2 | training_median | spearman | nan |
| realised_variance | 2 | zero | mae | 0.00051292952 |
| realised_variance | 2 | zero | rmse | 0.0098601164 |
| realised_variance | 2 | zero | mse | 9.7221895e-05 |
| realised_variance | 2 | zero | pearson | nan |
| realised_variance | 2 | zero | spearman | nan |

## Representation similarity

| Walk | Family | Variant | Linear CKA |
|---:|---|---|---:|
| 1 | contrastive | contrastive_lstm | 0.23041077 |
| 1 | contrastive | contrastive_transformer | 0.20098847 |
| 1 | byol | byol_lstm | 0.29607905 |
| 1 | byol | byol_transformer | 0.20884173 |
| 2 | contrastive | contrastive_lstm | 0.21512084 |
| 2 | contrastive | contrastive_transformer | 0.20437376 |
| 2 | byol | byol_lstm | 0.20758443 |
| 2 | byol | byol_transformer | 0.17086062 |
