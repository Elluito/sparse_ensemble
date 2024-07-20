import torch
import torch.nn.functional as F
import numpy as np


def calc_ece(confidences, accuracies, bins=15):
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    print("ECE {0:.2f} ".format(ece.item() * 100))

    return ece.item() * 100


def check_correctness(outputs, targets):
    total = 0
    correct = 0
    correct_soft_max = 0
    soft_max_outputs = F.softmax(outputs, dim=1)
    print("soft_max:{}".format(soft_max_outputs))
    _, predicted = torch.max(outputs.data, 1)
    soft_max_pred, predicted_soft_max = torch.max(soft_max_outputs.data, 1)
    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()

    correct_soft_max += predicted_soft_max.eq(targets.data).cpu().sum()

    return total, correct, correct_soft_max, predicted.eq(targets.data).cpu(), soft_max_pred


def correct_incorrect_max_prob(outputs, targets):
    _, predicted = torch.max(outputs.data, 1)
    soft_max_outputs = F.softmax(outputs, dim=1)
    soft_max_pred, predicted_soft_max = torch.max(soft_max_outputs.data, 1)
    correct_samples = predicted.eq(targets.data).cpu()
    incorrect_samples = ~correct_samples
    correct_max_prob = soft_max_pred[correct_samples]
    incorrect_max_prob = soft_max_pred[incorrect_samples]
    return correct_max_prob, incorrect_max_prob


def correct_incorrect_top_k_prob(outputs, targets, k=5):
    _, predicted = torch.max(outputs.data, 1)
    soft_max_outputs = F.softmax(outputs, dim=1)
    soft_max_pred, predicted_soft_max = torch.max(soft_max_outputs.data, 1)
    correct_samples = predicted.eq(targets.data).cpu()
    incorrect_samples = ~ correct_samples
    top_k_soft_max, top_k_soft_max_index = torch.topk(soft_max_outputs, k, dim=1).values, torch.topk(soft_max_outputs,
                                                                                                     k, dim=1).indices

    correct_topk_prob = top_k_soft_max[correct_samples, :]
    incorrect_topk_prob = top_k_soft_max[incorrect_samples, :]
    correct_topk_prob_index = top_k_soft_max_index[correct_samples, :]
    incorrect_topk_prob_index = top_k_soft_max_index[incorrect_samples, :]

    return correct_topk_prob, incorrect_topk_prob, correct_topk_prob_index, incorrect_topk_prob_index


def check_none_and_replace(accumulator, value):
    if accumulator is None:
        return value
    else:
        return torch.cat((accumulator, value))


def get_all_true_labels(dataloader, device):
    full_correct_labels = None

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        full_correct_labels = check_none_and_replace(full_correct_labels, y)
    return full_correct_labels


@torch.no_grad()
def get_correctness_dataloader(model, dataloader, device,topk=5):
    model = model.to(device)
    model.eval()
    full_accuracies = None
    full_confidences = None

    full_max_prob_correct = None
    full_max_prob_incorrect = None
    full_topk_prob_correct = None
    full_topk_prob_incorrect = None
    full_topk_prob_correct_index = None
    full_topk_prob_incorrect_index = None
    full_correct_labels = None

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        outputs = model(x)
        total, correct, correct_soft_max, accuracies, confidences = check_correctness(outputs, y)
        correct_maxprob, incorrect_maxprob = correct_incorrect_max_prob(outputs, y)
        correct_topk, incorrect_topk, correct_topk_index, incorrect_topk_index = correct_incorrect_top_k_prob(outputs,
                                                                                                              y,k=topk)
        full_accuracies = check_none_and_replace(full_accuracies, accuracies)
        full_confidences = check_none_and_replace(full_confidences, confidences)
        full_max_prob_correct = check_none_and_replace(full_max_prob_correct, correct_maxprob)
        full_max_prob_incorrect = check_none_and_replace(full_max_prob_incorrect, incorrect_maxprob)
        full_topk_prob_correct = check_none_and_replace(full_topk_prob_correct, correct_topk)
        full_topk_prob_incorrect = check_none_and_replace(full_topk_prob_incorrect, incorrect_topk)
        full_topk_prob_incorrect_index = check_none_and_replace(full_topk_prob_incorrect_index, incorrect_topk_index)
        full_topk_prob_correct = check_none_and_replace(full_topk_prob_correct_index, correct_topk_index)
        full_correct_labels = check_none_and_replace(full_correct_labels, y)

    return full_accuracies, full_confidences, full_max_prob_correct, full_max_prob_incorrect, full_topk_prob_correct, full_topk_prob_incorrect, full_topk_prob_correct_index, full_topk_prob_incorrect_index, full_correct_labels


if __name__ == '__main__':
    from torchvision.models import resnet18
    from torchvision.datasets import FakeData
    from torchvision import transforms

    # device = "cuda" if torch.cuda.is_available() else "cpu
    device = "cuda"
    model = resnet18()
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = FakeData(size=200, transform=transform_test)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=3, shuffle=False, num_workers=0)

    all_results = get_correctness_dataloader(model, testloader, device)

    print("ECE of fake data: {}".format(calc_ece(all_results[1], all_results[0])))

    print("Done")
