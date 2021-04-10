Description of files in a directory:
    prepare_dataset.py - preparing final dataset from initial and doing some analysis
    
    abstracts_training_xla.py - main training developed for training on 8 TPU cores, but properly working only on 1 core.
    
    abstracts_training_pl.py - main training reworked with pytorch-lightning, which allows properly train on 8 TPU cores.
    
    abstract_sample_gen.py - script for generating samples based on saved trained GPT2 model
    
    config.json - file containing training configuration
    
    colab_config.json - training configuration, when executing in colab
    
    GPT2-arXiv-tuning.pptx - PowerPoint presentation for the course work
