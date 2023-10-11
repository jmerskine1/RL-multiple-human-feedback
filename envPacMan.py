import numpy as onp
import jax.numpy as np
import jax
from config import environment as env
from config import parameters, setup, rl
from matplotlib import pyplot as plt
import library.utilities as ut



""" --------------------------------------------------------------------------
Ghost - class
"""
class ghost:
    def __init__(self):
        self.reset()
        self.dir = env['actions']['s']
        self.P_change_direction = 0.5       
        self.key = parameters['key'] 
        return
    
    def reset(self, x=[], y=[]):
        if x==[]:
            self.pos = env['ghost_init_pos']
        else:
            self.pos = np.array([x,y])
            
        self.prepos = self.pos
        return
    @profile
    def move(self):
        #check if the same movement is possible
        newPos = ut.newPos(self.pos, self.dir)
        cantMove = (newPos.all == self.pos.all)
        
        if cantMove or jax.random.uniform(self.key) < self.P_change_direction:

            newPos == self.pos.copy()
            
            while np.array_equal(newPos, self.pos):
                self.key, subkey = jax.random.split(self.key)
                self.dir = ut.randMove(subkey)
                newPos = ut.newPos(self.pos,self.dir)
                

        # actually move
        self.prepos = self.pos
        self.pos = newPos        
        return
    


""" --------------------------------------------------------------------------
Pacman - class
"""
class pacman:
    def __init__(self):
        # self.init_pos = env['pacman_init_pos']
        self.reset()
        self.key = parameters['key'] 
        return
    
    def reset(self, x=[], y=[]):
        if x==[]:
            self.pos = env['pacman_init_pos']
        else:
            self.pos = [x,y]
        self.prepos = self.pos
        return
    @profile
    def move(self, action):
        self.prepos = self.pos
        self.pos = ut.newPos(self.pos, action)  
        return
    
        
""" --------------------------------------------------------------------------
Pellets - class (Toplevel)
"""
class pellets:
    def __init__(self, pos_list):
        self.pos_list = np.array(pos_list)
        self.valid = np.array([ True for i in range(len(pos_list))])
        return
    
    def reset(self):
        self.valid = self.valid.at[:].set(True)
    
    def remaining_pellets(self):
        return self.pos_list[self.valid]
    
    def number_remaining_pellets(self):
        return np.sum(self.valid)
    
    def eaten(self, pos):
        ret = False
        for i in range(len(self.valid)):
            if all( np.array(pos) == self.pos_list[i] ) and self.valid[i]:
                self.valid[i] = False
                ret = True
        return ret
    
""" --------------------------------------------------------------------------
env - class (Toplevel)
"""
class environment:

    def __init__(self):
        self.pacman = pacman()
        self.ghost  = ghost()
        self.pellets = pellets([[3,3], [1,5]])        
        self.num_pellets = self.pellets.number_remaining_pellets()
        

        if setup['dispON']:
            self.fig, self.ax = plt.subplots(figsize = (8,8))
            self.ax.set_aspect(1.0)

            # Plot vertical lines
            for i in range(env['size']['X'] + 1):
                self.ax.plot([0, env['size']['X']],[i, i],  color='black')
            for i in range(env['size']['Y'] + 1):
                self.ax.plot([i, i], [0, env['size']['Y']], color='black')
            self.ax.set_xticks([])
            self.ax.set_yticks([])

            for obstacle in env['obstacles']:
                plt.scatter(obstacle[0],obstacle[1],s = 200,c='k')
            
            for pellet in env['pellets']:
                plt.scatter(pellet[0],pellet[1],s = 100,c='r')


            self.pacman_plot = self.ax.scatter(env['pacman_init_pos'][0],env['pacman_init_pos'][1],s=400,c='y')
            self.ghost_plot = self.ax.scatter(env['ghost_init_pos'][0],env['ghost_init_pos'][1],s= 400,c='b')
            plt.pause(0.05)
            
        return
    
    def reset(self, random=False):
        if random:
            xlim = env['size']['X']
            ylim = env['size']['Y']
            
            # pacman location
            while True:
                px = jax.random.randint(0, xlim)
                py = jax.random.randint(0, ylim)
                if self.map[px][py] != '#':
                    self.pacman.reset(px,py)
                    break
            # ghost location
            while True:
                gx = jax.random.randint(0, xlim)
                gy = jax.random.randint(0, ylim)
                if self.map[gx][gy] != '#' and (px!=gx or py!=gy):
                    self.ghost.reset(gx,gy)
                    break
        else:
            self.pacman.reset()
            self.ghost.reset()
            
        self.pellets.reset()        
        return
    
    @profile
    def step(self, action):
        rw = 0
        done = False

        # move pacman
        self.pacman.move(action)
        """
        while np.array_equal(self.pacman.pos,self.pacman.prepos):
            rw += -20
            self.pacman.key,subkey = jax.random.split(self.pacman.key)
            self.pacman.move(ut.randMove(subkey))        
        """
        # move ghost
        self.ghost.move()
        
        # collison?
        if np.array_equal(self.pacman.pos, self.ghost.pos) or (np.array_equal(self.pacman.prepos, self.ghost.pos) and np.array_equal(self.pacman.pos, self.ghost.prepos)):
            # Game over
            rw = -500
            done = True
        
        # eat pellet?
        eat = self.pellets.eaten(self.pacman.pos)
        if eat:
            rw += 10
            if self.pellets.number_remaining_pellets() == 0:
                # Clear Game
                rw += 500
                done = True
        else:
            rw -= 1
                
        # return [self.st2ob(), rw, done]
        return [self.st2ob(), rw, done]
        # pacman_pos,ghost_pos,ghost_dir,pellets_valid
    
    # generate display string
    def display(self):
        # self.fig.canvas.draw()
        self.pacman_plot.set_offsets(np.column_stack((self.pacman.pos[0],self.pacman.pos[1])))
        self.ghost_plot.set_offsets(np.column_stack((self.ghost.pos[0],self.ghost.pos[1])))

        plt.pause(0.05)

        display =False
        if display:
            disp = self.map.copy()

            p_pos = self.pellets.remaining_pellets()
            for i in range(self.pellets.number_remaining_pellets()):
                disp[p_pos[i][1]] = self.replaceChar(disp[p_pos[i][1]], '*', p_pos[i][0])

            disp[self.pacman.pos[1]] = self.replaceChar(disp[self.pacman.pos[1]], 'P', self.pacman.pos[0])
            disp[self.ghost.pos[1]]   = self.replaceChar(disp[self.ghost.pos[1]],   'G', self.ghost.pos[0])
                
            return disp
        else:
            return None
    

    
    # state to observation conversion
    @profile
    def st2ob(self):

        pPosIdx = (self.pacman.pos[0]) + (self.pacman.pos[1])*(env['size']['X']+1)     # Position of pacman
        gPosIdx = (self.ghost.pos[0]) + (self.ghost.pos[1])*(env['size']['X']+1)       # Position of ghost
        
        # gDirIdx = list(env['actions'].values()).index(self.ghost.dir.all())   #   int(np.where(np.all(np.array(list(env['actions'].values()))==self.ghost.dir,axis=1))[0])#list(env['actions'].keys()).index(self.ghost.dir) # Direction of ghost

        gDirIdx = ut.map_array_to_number(self.ghost.dir) 
        peltIdx = int(np.sum(np.array([(p+1)*self.pellets.valid[p] for p in np.arange(len(env['pellets']))]))) # Status of pellets
        
        return ut.index_to_element(rl['stateShape'],(pPosIdx,gPosIdx,gDirIdx,peltIdx))
    # def st2ob(self):

    #     return ut.st2ob(self.pacman.pos,self.ghost.pos,self.ghost.dir,self.pellets.valid)