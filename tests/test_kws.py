import os

from datetime import datetime

from blossom.models import MHAttKWS
from blossom.datasets import MHAttDataset
from blossom.learners import MHAttKWSLearner

def test_train():

    train_dataset = MHAttDataset(
        mode='train',
        root='./data/trigger/raw'
    )
    valid_dataset = MHAttDataset(
        mode='valid',
        root='./data/trigger/raw'
    )

    model = MHAttKWS(
        num_classes=2,
        in_channel=1,
        hidden_dim=128,
        n_head=4,
        dropout=0.1
    )

    learner = MHAttKWSLearner(model=model)
    learner.train(
        train_dataset=train_dataset,
        test_dataset=valid_dataset,
        batch_size=48,
        learning_rate=1e-4,
        eps=1e-8,
        betas=(0.9, 0.999),
        max_steps=10,
        n_epochs=1,
        shuffle=True,
        num_workers=8,
        view_model=True,
        save_path='./models',
        model_name='mhatt_model'
    )
    
# test_train()


def test_evaluate():

    test_dataset = MHAttDataset(
        mode='test',
        root='./data/trigger/raw'
    )

    model = MHAttKWS(
        num_classes=2,
        in_channel=1,
        hidden_dim=128,
        n_head=4,
        dropout=0.1
    )

    learner = MHAttKWSLearner(model=model)
    learner.load_model(model_path='./models/mhatt_model_best.pt')

    learner.evaluate(
        test_dataset=test_dataset,
        batch_size=48,
        num_workers=8,
        view_classification_report=True
    )

test_evaluate()


def test_inference():
    model = MHAttKWS(
        num_classes=2,
        in_channel=1,
        hidden_dim=128,
        n_head=4,
        dropout=0.1
    )

    learner = MHAttKWSLearner(model=model)
    learner.load_model(model_path='./models/mhatt_model_best.pt')

    now = datetime.now()

    output = learner.inference(input='data/trigger/raw/test/active/5f01c798_nohash_1.wav')

    print(output)

    print(f"\nInference time: {(datetime.now() - now) * 1000} ms")


test_inference()