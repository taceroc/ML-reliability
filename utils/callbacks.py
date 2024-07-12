class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    
class BestAccuracy:
    def __init__(self, min_delta=0):
        self.min_delta = min_delta
        self.max_f1 = 0

    def best_f1(self, f1, epoch):
        if f1 > 0:
            if f1 > self.max_f1:
                self.max_f1 = f1
                print("close", epoch)
                return True
            elif f1 < (self.max_f1 + self.min_delta):
                return False
        else:
            return False
        