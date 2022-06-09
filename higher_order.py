import torch
from torchvision import models
import copy
import higher

def _objective(model, poison_x, x, y):
    adv_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, adv_model.parameters()), lr=0.1)
    # Wrap the model into a meta-object that allows for meta-learning steps via monkeypatching:
    with higher.innerloop_ctx(adv_model, optimizer, copy_initial_weights=False) as (fmodel, fopt):
        for _ in range(1):
            #the first step : x --> model weights
            outputs = fmodel(torch.cat([x,poison_x]))
            #calculating a loss 1 (entropy here) which depends on the prediction of the model for posioned input
            poison_loss = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1).mean(0)
            #updating the model weights to minimize the entropy of predictions on the poisoned input
            fopt.step(poison_loss)

    #calculating auxilliary loss (l2) using new model weights
    new_preds = fmodel(x)
    #auxilliary loss with the updated model parameteres
    new_loss = torch.nn.CrossEntropyLoss()(new_preds, y)
    return new_loss

def get_poison_delta(model, input, y):
    #initializing posion examples to zeros for simplicity (posion_delta represents the poisoned_input that will be added to the dataset)
    poison_delta = torch.zeros_like(input, requires_grad=True)
    poison_optimizer = torch.optim.SGD([poison_delta], lr=0.1, momentum=0.9, weight_decay=0)
    num_poison_iters = 10
    for _ in range(num_poison_iters):
        poison_optimizer.zero_grad()
        # Gradient step
        loss = _objective(model, poison_delta, input, y)
        #d(l2)/d(posion_delta)
        #*****************THE SECOND ORDER GRADIENTS******************
        poison_delta.grad, = torch.autograd.grad(loss, poison_delta, retain_graph=False, create_graph=False, only_inputs=True)
        # Optim step
        poison_optimizer.step()
        # Projection step (omitted for simplicity)
    poison_delta.requires_grad = False
    return poison_delta

def main():
    model = models.resnet18().cuda()
    print("Starting")
    batch_size, dims, pix = 100, 3, 32
    # using a random input just for a minimalistic example to test gradient flow
    input = torch.rand(batch_size, dims, pix, pix).cuda()
    y = torch.randint(0, 10, (batch_size,)).cuda()

    poisoned_image = get_poison_delta(model, input, y)

    # print(poisoned_image)
if __name__ == "__main__":
    main()

