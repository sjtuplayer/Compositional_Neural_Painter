import random
class memory:
    def __init__(self):
        self.states=[]
        self.l=1000
    def append(self,state,step=None,alpha_canvas=None,device='cpu'):
        if device=='cpu':
            state=state.cpu()
        if alpha_canvas is None:
            self.states.append((state.detach(),step))
        else:
            self.states.append((state.detach(), step,alpha_canvas))
        if len(self.states)>self.l:
            self.states.pop(random.randint(0,self.l-1))
    def random_reset(self):
        random.shuffle(self.states)
    def pop(self):
        return self.states.pop()
    def empty(self):
        if len(self.states)==0:
            return True
        else:
            return False
class memory2:
    def __init__(self):
        self.states=[]
    def append(self,state,old_canvas,step=None,alpha_canvas=None,device='cpu'):
        if device=='cpu':
            state=state.cpu()
        self.states.append((state.detach(),old_canvas,step))
    def random_reset(self):
        random.shuffle(self.states)
    def pop(self):
        return self.states.pop()
    def empty(self):
        if len(self.states)==0:
            return True
        else:
            return False