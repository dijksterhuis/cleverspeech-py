# config/arguments_parser

Here's a list of available command line options when you want to run one of the scripts in
[cleverspeech/scripts](../cleverspeech/scripts)

|argument|defaults|description|
|---|---|---|
|`gpu_device`|`0`| which GPU device ID (first, second, third, etc.) to use|
|`batch_size`|`1`| how many examples to generate at the same time|
|`skip_n_batch`|`0`| if you have run the script on N batches already, you can skip ahead to batch N|
|`learning_rate`|`10 / 2**15`| default learning rate for loaded optimisers |
|`nsteps`|`10000`| how many steps of optimisation to try and generate adversarial examples |
|`decode_step`|`100`| when to stop and check if an example returns a successful decoding |
|`restart_step`|`2500`| when to randomise a perturbation if they've never been successful (requires the `*WithRandomRestarts` procedure type) |
|`constraint_update`|`geom`| stale config -- do not use |
|`rescale`|`0.9`| how much to reduce bounds by during clipped gradient descent |
|`audio_indir`|`./samples/all/`| where your input data lives, note that each wav needs an additional `.json` file |
|`outdir`|`./adv/`| where the output results data is written to, note that the `S3` file writer treats this as the s3 bucket location |
|`targets_path`|`./samples/cv-valid-test.csv`| where the transcriptions `.csv` file lives |
|`max_examples`|`100`| how many examples to try to find perturbations for |
|`max_targets`|`2000`| maximum number of transcription to load as possible targets, note this isn't needed anymore |
|`max_audio_file_bytes`|`120000`| sets an upper bound on audio file size -- larger files can lead to OOM errors |
|`pgd_rounding`|`0`| unused, please ignore |
|`delta_randomiser`|`0`| the amplitude of the random uniform noise when initialising perturbations |
|`beam_width`|`500`| beam width for beam search decoders (has no effect on greedy search decoders) |
|`random_seed`|`4568`| default random seed for all random ops (this is mozilla/DeepSPeech's default) |
|`align`|`mid`| default path to select when doing offline path based attacks |
|`align_repeat_factor`|`None`| unused, please ignore |
|`decoder`|`batch`| which decoder type to use (default is mozilla/DeepSpeech's batched beam search decoder with language model) |
|`writer`|`local_latest`| where and when to write out results data |
