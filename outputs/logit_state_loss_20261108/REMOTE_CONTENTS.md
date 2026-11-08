# Remote archive scope

This commit publishes the selected group_mean_new implementation, its self-contained runner, unchanged configuration, tests, all 11 raw quality results, input manifests, per-layer state records, timing, GPU monitor records, and source hashes. It also publishes the cross-candidate summary tables and final status from the seven-method research campaign.

Large diagnostic tensors, experimental alternative implementations, and their complete run directories remain in the verified local archives outputs/logit_state_loss_20261108 and outputs/logit_rgb_calibration_20261108 in the main workspace. REPORT.md describes that full local archive; its links to other candidate run directories are not a claim that those directories are included in this selected-method commit. archive_manifest.json describes the full local archive, not a Git tree. No checkpoint or dataset is bundled.

The current runner only changes the default configuration filename to logit_group_mean_eth3d.json; its JSON content is identical to the configuration used for the measurements. Its legacy --variant centered050 argument is retained for command compatibility; the installed method is group_mean_logit_cumulant050, asserted in the layer records.
