# Dataset audit 

- WAVs in archive: 2,800.
- Each selected corpus has 14 speaker/emotion directories: OAF and YAF across seven emotion classes, 200 files per directory 
- Folder labels are case-insensitive. `OAF_Pleasant_surprise` and `YAF_pleasant_surprised` map to the common `ps` label.
- The filename has a speaker prefix, target word, and emotion suffix (e.g. `YAF_youth_ps.wav`). The emotion suffix must be stripped to create text input; it must never be used as a feature.
- The transcript is reconstructed as “say the word [target word]” per TESS's carrier-phrase design. The audio should be checked to confirm the carrier phrase used in this copy.

These are data inventory observations, not model performance results.

