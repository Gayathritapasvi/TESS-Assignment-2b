# Multimodal Emotion Recognition on TESS

PyTorch reference implementation for speech-only, text-only, and early-fusion emotion recognition using the Toronto Emotional Speech Set (TESS).

## Dataset setup
Download and extract the Kaggle dataset: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess . Place the folders containing WAV files under `data/` (or provide another directory with `--data-dir

It contains 2,800-file copy of TESS. 

## Install and run
```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python models/speech_pipeline/train.py --data-dir data
python models/speech_pipeline/test.py --data-dir data
python models/text_pipeline/train.py --data-dir data
python models/text_pipeline/test.py --data-dir data
python models/fusion_pipeline/train.py --data-dir data
python models/fusion_pipeline/test.py --data-dir data
```
Each pipeline writes its best checkpoint, fixed split CSVs, metrics JSON, confusion matrix CSV, and misclassified examples beneath `Results/<mode>/`. The split is stratified by emotion (70/15/15) and fixed by seed 42. 

## Design
Speech: mono 16 kHz audio, silence trimming, log-Mel spectrogram (64 mel bins), compact 1D convolutional encoder and classifier. Text: token embeddings and bidirectional GRU. Fusion: concatenate 128-dimensional audio and 192-dimensional text representations, followed by an MLP classifier. Validation loss with early stopping selects the checkpoint. Test data is used only for final evaluation.

## Important evaluation caveat
TESS is a small, acted corpus from two speakers; examples share lexical content and recording conditions. A random file-level split can place repetitions of the same speaker/word in both train and test, so it estimates performance on this corpus rather than robust unseen-speaker generalization. A speaker-held-out check is desirable, but with two speakers it has only two folds and is unstable. Text-only results may be weak because the transcript is repeated across emotions and emotional prosody is absent from text

## Results/report
The `Results/` folder contains measured metrics, confusion matrices, misclassified examples, and embedding plots from the supplied run. The report draft discusses the outcomes, actual errors, qualitative cluster patterns, and evaluation caveats. 





### Regenerate learned-representation plots
After training all three models, run:
```bash
python models/common/plot_embeddings.py --data-dir data
```
This creates held-out t-SNE visualizations under `Results/plots/`. Confusion-matrix plots and score tables are also included for the supplied run; rerun each test script after retraining to refresh per-model metrics.

## Supplied run
The included `Results/` folder contains a completed reference run with the seed-42 stratified 70/15/15 split. Accuracy was 99.52% speech-only, 12.62% text-only, and 95.24% fusion. The unusually strong speech score should be interpreted cautiously: this is a random file-level split of a small two-speaker acted corpus, not a speaker-independent test.

