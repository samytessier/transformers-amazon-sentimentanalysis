#file for loading/saving checkpoints (trained models)
import torch

def save_checkpoint(filepath, model):
    checkpoint = {
        'height': model.height,
        'width': model.width,
        'channels': model.channels,
        'classes': model.classes,
        'dropout': model.dropout_rate,
        'state_dict': model.state_dict()}

    torch.save(checkpoint, filepath)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = ImageClassifier(
        height=checkpoint['height'],
        width=checkpoint['width'],
        channels=checkpoint['channels'],
        classes=checkpoint['classes'],
        dropout=checkpoint['dropout'])
    model.load_state_dict(checkpoint['state_dict'])
    return model
