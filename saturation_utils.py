from delve import SaturationTracker
import torch.nn as nn
from sparse_ensemble_utils import test


def calculate_train_eval_saturation_solution(net, trainloader, testloader, save_folder, file_name_sufix, seed, device):
    csv_tracker = SaturationTracker("{}/{}_seed_{}/".format(save_folder, file_name_sufix, seed), save_to="csv",
                                    modules=net,
                                    device=device, average_sat=True)

    net.train()

    ##### First one epoch of the training data

    criterion = nn.CrossEntropy()
    correct = 0
    total = 0
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)

        # shape_inputs = inputs.shape[0]
        # print("shape_inputs {}".format(shape_inputs))
        # assert shape_inputs[-1] == 360, "Shape inputs: {} expected shape inputs to be {}".format(shape_inputs, 360)
        # break
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # print(100. * correct / total, correct, total)
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        # print(
        #     'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        if batch_idx % 10 == 0:
            print(
                'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    #### Second one of the testing data

    test_accuracy = test(net, True if device == "cuda" else False, testloader, verbose=0)

    ################################## Then save ####################

    csv_tracker.add_scalar("train accuracy", (correct / total) * 100)
    csv_tracker.add_scalar("test accuracy", test_accuracy)
    csv_tracker.add_saturations()
    csv_tracker.close()



def main():
    pass

if __name__ == '__main__':
    main()