import os
import torch

class YyxAgentBase():
    '''
    motivation: unify all save_nets() in all methods
    '''
    def __init__(self):
        pass

    def save_nets(self, dir_name, iter=0, is_newbest=False):
        if not os.path.exists(dir_name + '/Models'):
            os.mkdir(dir_name + '/Models')
        prefix = 'best' if is_newbest else str(iter)
        model_dir = os.path.join(dir_name, 'Models')
        # Save the full module state so custom agents can restore all of their submodules.
        torch.save(self.state_dict(), os.path.join(model_dir, prefix + '_agent.pt'))
        torch.save(self.actors.state_dict(), os.path.join(model_dir, prefix + '_actor.pt'))
        if self.input_args.algo.startswith('G2ANet'):
            torch.save(self.g2a_embed_hard_net.state_dict(), os.path.join(model_dir, prefix + '_g2aAtt.pt'))

        # print('RL saved successfully')

    def load_nets(self, dir_name, iter=0, best=False):
        prefix = 'best' if best else str(iter)
        model_dir = os.path.join(dir_name, 'Models')
        agent_path = os.path.join(model_dir, prefix + '_agent.pt')
        if os.path.exists(agent_path):
            self.load_state_dict(torch.load(agent_path, map_location=self.device))
            return

        # Backward-compatible fallback for older checkpoints.
        self.actors.load_state_dict(torch.load(os.path.join(model_dir, prefix + '_actor.pt'), map_location=self.device))
        if self.input_args.algo.startswith('G2ANet'):
            self.g2a_embed_hard_net.load_state_dict(torch.load(os.path.join(model_dir, prefix + '_g2aAtt.pt'), map_location=self.device))

