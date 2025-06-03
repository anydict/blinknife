import librosa
import numpy as np
from librosa.feature import melspectrogram


if __name__ == '__main__':
    file_name = 'da'

    y, sr = librosa.load(f'tmp/{file_name}.wav', duration=1)
    mel_spec = melspectrogram(y=y, sr=sr)
    print(mel_spec.max(), mel_spec.min())

    db_ms = librosa.power_to_db(mel_spec, ref=np.max) * -1
    int_db_ms = db_ms.astype(np.uint8)

    int_db_ms[np.where(db_ms > 70)] = 255
    int_db_ms[np.where(db_ms < 70)] = 150
    int_db_ms[np.where(db_ms < 30)] = 0

    # db_ms[np.where(db_ms >= -30)] = 255
    # db_ms[np.where(db_ms >= -30)] = 255

    print(db_ms.max(), db_ms.min())

    # db_ms = db_ms.astype(np.uint8)
    # db_ms =db_ms.astype(np.uint8)

    # db_ms[np.where(db_ms > 200)] = 255
    #
    from PIL import Image, ImageOps

    img = ImageOps.flip(Image.fromarray(int_db_ms, 'L'))
    img.save(f"images/{file_name}.png")


    #
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    img = librosa.display.specshow(int_db_ms, x_axis='time',
                                   y_axis='mel', sr=sr,
                                   fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    fig.show()
