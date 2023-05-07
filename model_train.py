import torch
import time
import model_test


def train_output_validation(model, validate_loader, criterion, running_loss, step_number,
                            start_text="",
                            training_text="Training Loss",
                            validation_loss_text="Validation Loss",
                            validation_accuracy_text="Validation Accuracy",
                            last_length=0,
                            device_flag="cpu"):

    valid_text = "\r" + start_text + "Validating..."
    print(valid_text + ' ' * (last_length - len(valid_text)+1), end="")

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        validation_loss, accuracy = model_test.validation(model, validate_loader, criterion, device_flag=device_flag)

    valid_text = f"\r{start_text}" \
                 + "{}: {:.4f}... ".format(training_text, running_loss / step_number) \
                 + "{}: {:.4f}... ".format(validation_loss_text, validation_loss / len(validate_loader)) \
                 + "{}: {:.4f}".format(validation_accuracy_text, accuracy / len(validate_loader))
    print(valid_text + ' ' * (last_length - len(valid_text)+1))


def train_classifier(model, train_loader, validate_loader, optimizer, criterion, device_flag="cpu", epochs=10,
                     validate_steps=100, validate_stepped=True, validate_epoch=False, validate_end=False):
    steps = 0

    model.to(device_flag)

    running_total = len(train_loader.dataset)

    start_time = time.time()
    running_loss = 0

    if validate_end and validate_epoch:
        print("Not validating the end, as all Epochs will be validated.")
        validate_end = False

    if epochs <= 0:
        print("Must have at least 1 epoch")
        return 0

    total_running_loss = 0
    model.train()

    iter_text = ""
    for e in range(epochs):
        epoch_text = f"[{e + 1}/{epochs}]"

        running_done = 0

        epoch_start = time.time()

        for images, labels in iter(train_loader):
            iter_text = "{} {} / {} - {:.3f}% complete, {} Batches completed.".format(epoch_text, running_done,
                                                                                      running_total,
                                                                                      (
                                                                                                  running_done / running_total) * 100,
                                                                                      steps)
            print(f"\r{iter_text}", end="")

            steps += 1

            images, labels = images.to(device_flag), labels.to(device_flag)

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_running_loss += loss.item()

            running_done += images.size(0)

            if validate_stepped:
                # If not on last step, and on print step
                if (steps != len(train_loader)) and steps % validate_steps == 0:
                    # If validating epoch, then mention that stepped validate has been
                    # pushed back.
                    if validate_epoch and steps % epochs == 0:
                        print(f"\r{epoch_text}Stepped validation intersected with Epoch validation, trying again in "
                              f"another {validate_steps} batches...")
                    else:
                        train_output_validation(model, validate_loader, criterion, running_loss, validate_steps,
                                                start_text=f"{epoch_text} Batch: {steps}... ",
                                                training_text="Training Loss since last stepped validation",
                                                last_length=len(iter_text),
                                                device_flag=device_flag)
                        running_loss = 0
                        model.train()

        time_now = time.time()
        finish_text = epoch_text + " Epoch {} completed on Batch {} in {:.4f} seconds ({:.4f} in total) ".format(e + 1,
                                                                                                                 steps,
                                                                                                                 time_now - epoch_start,
                                                                                                                 time_now - start_time)
        if validate_epoch:
            train_output_validation(model, validate_loader, criterion, total_running_loss, steps,
                                    start_text=finish_text,
                                    last_length=len(iter_text),
                                    device_flag=device_flag)
            model.train()
        else:
            print(f"\r{finish_text}{' ' * (len(iter_text) - len(finish_text))}\n")

    print("\nTraining Complete in {:.4f} seconds.".format(time.time() - start_time))
    if validate_end:
        print(f"\nValidation of all {steps} Batches:")
        train_output_validation(model, validate_loader, criterion, total_running_loss, steps,
                                last_length=len(iter_text),
                                device_flag=device_flag)
        model.train()

    # The number of batches total
    return steps
