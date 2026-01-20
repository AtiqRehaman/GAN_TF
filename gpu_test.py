import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth set True for gpu {gpu}")
            
    except Exception as e:
        print(e)
        
else:
    print("No GPU found!")
    
