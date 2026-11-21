# Multimodal Emotion Recognition on TESS — Assignment Report

## A. Architecture decisions

### Dataset and preprocessing
It  contains 2,800 WAV files: 2 speakers (OAF/YAF), 7 emotions, and 200 recordings per speaker-emotion folder. Emotion labels are parsed from the directory name, normalized to seven classes, and never supplied as input. Filenames also end in the emotion code, which is stripped before deriving text. The utterance transcript is reconstructed as “Say the word [target word]” from the target word in the filename. This matches the TESS carrier-phrase format, but transcript construction should be stated as a limitation.

Audio is loaded as mono, resampled to 16 kHz, silence-trimmed, capped at four seconds, and represented as a 64-bin log-Mel spectrogram. Frame sequences are padded/cropped to 400 frames for batching. Text is lowercased, whitespace-tokenized, and mapped to a vocabulary built from the dataset.

### Models
- **Speech:** two 1-D temporal convolution layers over log-Mel frames, global temporal pooling, a 128-dimensional learned representation, and a seven-class linear head. This compact model learns emotion-related acoustic patterns while limiting parameters for the small dataset.
- **Text:** trainable 64-dimensional token embeddings, a bidirectional GRU with 96 hidden units per direction, mean pooling, and a seven-class linear head. The vocabulary is small, so a compact recurrent model avoids a large pretrained model on a small corpus.
- **Fusion:** concatenate the 128-dimensional audio vector and 192-dimensional text vector, followed by an MLP classifier. This is early feature-level fusion.

All models use cross-entropy and AdamW. A fixed seed (42) and stratified file-level split produce 70% train, 15% validation, 15% test. Validation loss selects the checkpoint (early stopping); the held-out test set is evaluated once after training.

## B. Experiments and results

Measured on the same held-out set (420 examples; 60 per class):

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Speech-only | 0.9952 | 0.9952 | 0.9952 |
| Text-only | 0.1262 | 0.0774 | 0.0774 |
| Multimodal fusion | 0.9524 | 0.9530 | 0.9530 |

Detailed metrics and per-class confusion counts are in `Results/{speech,text,fusion}/test_metrics.json` and `confusion_matrix.csv`; comparison is in `Results/comparison.csv` and `.md`. Confusion-matrix figures are in `Results/plots/`. The embedding t-SNE figures visualize learned representations for the held-out samples.

## C. Analysis

### Easiest and hardest classes
Speech-only recall is 1.00 for angry, fear, happy, neutral, and sad; disgust and pleasant surprise are the lowest at 0.983 (59/60 each). For fusion, disgust, fear, neutral, and sad have recall 1.00; pleasant surprise is lowest at 0.85 (51/60). Text-only has recall 0 for fear, neutral, and sad, and 0.55 for happy. This is consistent with the limited lexical information: nearly identical carrier phrases/words recur across the emotions, while the emotional signal is primarily in vocal delivery.

### Does fusion help?
No, not in this run: fusion accuracy (0.9524) is below speech-only (0.9952), while text-only is near the seven-class chance level (1/7 ≈ 0.143) and has macro F1 0.0774. The text branch offers little independent emotion signal here, and adding a weak modality can make optimization harder. Fusion should not be claimed to improve performance based on these measurements. 
### Actual error examples
1. `OAF_thought_disgust.wav`: speech model predicted pleasant surprise instead of disgust.
2. `OAF_hall_ps.wav`: speech model predicted happy instead of pleasant surprise.
3. `OAF_love_ps.wav`: fusion model predicted disgust instead of pleasant surprise.
4. `OAF_raid_happy.wav`: fusion model predicted disgust instead of happy.
5. `YAF_week_disgust.wav`: text model predicted angry instead of disgust.

These errors are compatible with ambiguity between acted expressive categories and the inability of text alone to encode prosody.
### Representation visualizations
- Speech temporal representation: `Results/plots/speech_embedding_tsne.png`
- Text contextual representation: `Results/plots/text_embedding_tsne.png`
- Fusion representation: `Results/plots/fusion_embedding_tsne.png`

The figures use t-SNE on the learned vectors of held-out test items: the pooled CNN audio vector, the pooled BiGRU text vector, and the concatenated audio-text vector before the fusion MLP. In the speech plot, emotion groups are visually well-separated, with a small amount of overlap/outliers (consistent with the two speech errors). The text plot is heavily mixed: examples of different labels occupy the same broad regions, consistent with repeated lexical inputs and weak text-only performance. The fusion plot forms several distinct groups, but happy and pleasant-surprise points overlap and some angry examples lie near the disgust region; the fusion errors show related confusions. These are qualitative patterns only: t-SNE is stochastic/nonlinear, axes and distances are not calibrated, and apparent separation does not replace the held-out quantitative scores.

### Limitations and interpretation
This is a small acted dataset with only two speakers. The file-level random split places samples from both speakers and repeated lexical items in train, validation, and test, so the very high speech score does not demonstrate generalization to unseen speakers or natural speech. A speaker-held-out evaluation would be preferable, but with just two speakers it is only two unstable folds and should be reported separately. Text is reconstructed using the filename target and carrier phrase; its poor performance is therefore informative about the dataset’s limited text signal, but should not be generalized to transcripts that explicitly describe emotional states. The test set was used for reported metrics only after model selection; no hyperparameter tuning was done on it. The code/results/report package is prepared locally
