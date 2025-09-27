# Object-AVEdit: An Object-level Audio-Visual Editing Model

## 📚 Object-AVEdit

There is a high demand for audio-visual editing in video post-production and the film making field. While numerous models have explored audio and video editing, they struggle with object-level audio-visual operations. Specifically, object-level audio-visual editing requires the ability to perform object addition, replacement, and removal across both audio and visual modalities, while preserving the structural information of the source instances during the editing process. In this paper, we present \textbf{Object-AVEdit}, achieving the object-level audio-visual editing based on the inversion-regeneration paradigm. To achieve the object-level controllability during editing, we develop a word-to-sounding-object well-aligned audio generation model, bridging the gap in object-controllability between audio and current video generation models. Meanwhile, to achieve the better structural information preservation and object-level editing effect, we propose an inversion-regeneration holistically-optimized editing algorithm, ensuring both information retention during the inversion and better regeneration effect. Extensive experiments demonstrate that our editing model achieved advanced results in both audio-video object-level editing tasks with fine audio-visual semantic alignment. In addition, our developed audio generation model also achieved advanced performance. **Demo--more video samples in our [project page](https://gewu-lab.github.io/Object_AVEdit-website/)!**

## 🛠️ Setup

```bash
cd Object-AVEdit
# setup base environment
conda env create -f environment.yml
conda activate avedit
# install flash-attn3
git clone https://github.com/Dao-AILab/flash-attention 
cd flash-attention/hopper
python setup.py install
# if run training
pip install colossalai --no-deps
```



## 🚀 Prepare weights

**Audio generation weight.**

Download from [here](https://huggingface.co/FYQ12138/AVCE-P2P/tree/main) and place it in \audio_weight.

**Video generation weight.**

Download from [here](https://huggingface.co/genmo/mochi-1-preview/tree/main) and place it in \video_weight.

**Other weights needed.**

T5-large text encoder from [here](https://huggingface.co/google-t5/t5-large).

AudioLDM Mel vocoder from [here](https://huggingface.co/cvssp/audioldm-m-full/tree/main).



## 🚀 Usage

We provide three types of tasks:

1. Audio Generation Task.
2. Audio Editing Task.
3. Video Editing Task.



### Audio Generation

Fisrt, change the path in /audio_generation_model/generation.py

```bash
cd audio_generation_model

python generation.py --prompt "dog bark" --GPU_num=0 
```

### Audio Editing

Fisrt, change the path in /audio_edit_part/audio_edit_main.py

```bash
cd /audio_edit_part

python audio_edit_main.py \
--input_audio "/demo_data/replace_data/21.wav" \
--source_prompt="Several brown cows are standing in a green alpine meadow under the tall, snowy mountains." \
--target_prompt="Several brown horses are standing in a green alpine meadow under the tall, snowy mountains." \
--edit_type="Replacement" \
--GPU_num=0 \
--word="cows" \
--seed=42 \
--CA=50 \
--SA=50 \
--threshold=0.1
```

--word parameter means the target object that you want to edit. In replacement and removal tasks, the --word should be set to the source object to be edtied, and in addition task, --word shouldnot be set.

--threshold parameter means the mask process. Biger threshold means less change.

--CA means the cross attention control steps. 

--SA means the cross attention control steps. 

Biger CA or SA means more control steps. In removal task, SA is often set to 0.



### Video Editing

Fisrt, change the path in /video_edit_part/config.py.

```bash
cd /audio_edit_part

python video_edit_main.py \
--input_video "/demo_data/replace/6.mp4" \
--source_prompt="Several brown cows are standing in a green alpine meadow under the tall, snowy mountains." \
--target_prompt="Several brown horses are standing in a green alpine meadow under the tall, snowy mountains." \
--edit_type="Replacement" \
--GPU_num=0 \
--word="cows" \
--seed=42 \
--CA=37\
--SA=37 \
--threshold=0.05 
```

--word parameter means the target object that you want to edit. In replacement and removal tasks, the --word should be set to the source object to be edtied, and in addition task, --word shouldnot be set.

--threshold parameter means the mask process. Biger threshold means less change.

--CA means the cross attention control steps. 

--SA means the cross attention control steps. 

Biger CA or SA means more control steps. In removal task, SA is often set to 0.



## ☀️ Acknowledgements

Our project is partly based on the [Mochi-1](https://huggingface.co/genmo/mochi-1-preview), [AudioLDM](https://huggingface.co/cvssp/audioldm-m-full/tree/main) model. We would like to thank the authors for their excellent work! ❤️









