import copy


def train_model(model, criterion, optimizer, num_epochs=100):

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

