import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patheffects as pe

""" --------------------------------------------------------------------------
Ghost - class
"""
class ghost:
    def __init__(self, x, y, map):
        self.init_pos = [x, y]
        self.reset()

        self.dir = 's'
        self.map = map
        
        self.P_change_direction = 0.5        
        return
    
    def reset(self, x=[], y=[]):
        if x==[]:
            self.pos = self.init_pos
        else:
            self.pos = [x,y]
            
        self.prepos = self.pos       
        return
    
    def move(self):
        #check if the same movement is possible
        newPos = self.newPos(self.pos, self.dir)
        cantMove = (newPos == self.pos)
        
        if cantMove or np.random.rand() < self.P_change_direction:
            newPos == self.pos.copy()
            while newPos == self.pos:
                r = np.random.rand()
                if  r < 1/4:
                    self.dir = 'n'
                elif r < 2/4:
                    self.dir = 's'
                elif r < 3/4:
                    self.dir = 'w'
                else:
                    self.dir = 'e'
                
                newPos = self.newPos(self.pos, self.dir)
            
        # actually move
        self.prepos = self.pos
        self.pos = newPos        
        return
    
    def newPos(self, currPos, dir):
        
        newPos = currPos.copy()      
        if dir == 'n':
            newPos[1] -= 1
        elif dir == 's':
            newPos[1] += 1
        elif dir == 'w':
            newPos[0] -= 1
        elif dir == 'e':
            newPos[0] += 1
        if self.map[newPos[1]][newPos[0]] == '#':
            newPos = currPos           
        return newPos

""" --------------------------------------------------------------------------
Pacman - class
"""
class pacman:
    def __init__(self, x, y, map):
        self.init_pos = [x, y]
        self.reset()

        self.map = map
        return

    def reset(self, x=[], y=[]):
        if x==[]:
            self.pos = self.init_pos
        else:
            self.pos = [x,y]
        self.prepos = self.pos
        self.last_dir = None   # last direction that produced actual movement
        return

    def move(self, action):
        self.prepos = self.pos
        new_pos = self.newPos(self.pos, action)
        if new_pos == self.pos:
            # Chosen action hits a wall – try to continue in the last
            # successful direction instead of standing still.
            if self.last_dir is not None:
                fallback = self.newPos(self.pos, self.last_dir)
                if fallback != self.pos:
                    new_pos = fallback
        else:
            self.last_dir = action   # record direction that caused movement
        self.pos = new_pos
        return
    
    def newPos(self, currPos, dir):
        
        newPos = currPos.copy()
        if dir == 'n':
            newPos[1] -= 1
        elif dir == 's':
            newPos[1] += 1
        elif dir == 'w':
            newPos[0] -= 1
        elif dir == 'e':
            newPos[0] += 1
    
        if self.map[newPos[1]][newPos[0]] == '#':
            newPos = currPos    
        return newPos

""" --------------------------------------------------------------------------
Pellets - class (Toplevel)
"""
class pellets:
    def __init__(self, pos_list):
        self.pos_list = np.array(pos_list)
        self.valid = np.array([ True for i in range(len(pos_list))])
        return
    
    def reset(self):
        self.valid[:] = True
    
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
class env:

    def __init__(self, size='small'):
        self.map = list()
        if size == 'small':
            self.map.append('#######')
            self.map.append('#     #')
            self.map.append('# ### #')
            self.map.append('# #   #')
            self.map.append('# ### #')
            self.map.append('#     #')
            self.map.append('#######')
                                    
            self.pacman = pacman(1,1, self.map)
            self.ghost  = ghost(5,5, self.map)
            self.pellets = pellets([[3,3], [1,5]])
            
        elif size == 'medium':        
            self.map.append('###########')
            self.map.append('#         #')
            self.map.append('# ### ### #')
            self.map.append('# #   # # #')
            self.map.append('# # #   # #')
            self.map.append('# ### ### #')
            self.map.append('#         #')
            self.map.append('###########')
                                    
            self.pacman = pacman(1,1, self.map)
            self.ghost  = ghost(9,6, self.map)
            self.pellets = pellets([[3,4],[7,3],[1,6],[9,1]])
        
        elif size == 'medium_sparse':        
            self.map.append('###########')
            self.map.append('#         #')
            self.map.append('# ### ### #')
            self.map.append('# #   # # #')
            self.map.append('# # #   # #')
            self.map.append('# ### ### #')
            self.map.append('#         #')
            self.map.append('###########')
                                    
            self.pacman = pacman(1,1, self.map)
            self.ghost  = ghost(9,6, self.map)
            self.pellets = pellets([[3,4],[7,3]])

        else:
            raise ValueError(f"invalid environment size is specified: {size}")

        self.map_size_x = len(self.map[0]) - 2
        self.map_size_y = len(self.map) - 2
        self.num_pellets = self.pellets.number_remaining_pellets()

        # Compute inner-wall obstacles from map (any '#' not on the outer border)
        self.obstacles = []
        for y in range(1, len(self.map) - 1):
            for x in range(1, len(self.map[0]) - 1):
                if self.map[y][x] == '#':
                    self.obstacles.append([x, y])

        # Load sprite images for plotting
        self.pacman_img = image.imread('./sprites/pacman.png')
        self.ghost_img  = image.imread('./sprites/ghost.png')
        self.pellet_img = image.imread('./sprites/cherry.png')
        self.walls = self.add_walls()
        return
    
    def reset(self, random=False, pellet_random=False):
        if random:
            xlim = len(self.map[0])
            ylim = len(self.map)
            
            # pacman location
            while True:
                px = np.random.randint(0, xlim)
                py = np.random.randint(0, ylim)
                if self.map[py][px] != '#':
                    self.pacman.reset(px,py)
                    break
            # ghost location
            while True:
                gx = np.random.randint(0, xlim)
                gy = np.random.randint(0, ylim)
                if self.map[gy][gx] != '#' and (px!=gx or py!=gy):
                    self.ghost.reset(gx,gy)
                    break
        else:
            self.pacman.reset()
            self.ghost.reset()
            
        self.pellets.reset()
        if pellet_random:
            # random pellet enable/disable
            while True:
                valid = np.random.randint(0, 2, len(self.pellets.valid)) == 1
                if np.any(valid):
                    break
            self.pellets.valid = valid
        return self.st2ob()
    
    def nStates(self):
        return (self.map_size_x*self.map_size_y) * 4 * (2**self.num_pellets) * (self.map_size_x*self.map_size_y) # Ghost pos. x Ghost direction x pellets x Pacman pos.
        
    def action_list(self):
        return ['n', 's', 'e', 'w']
    
    def step(self, action):
    
        # init. return parameters
        rw = 0
        done = False

        # move pacman
        self.pacman.move(action)
        
        # move ghost
        self.ghost.move()
        
        # collison?
        if self.pacman.pos == self.ghost.pos or (self.pacman.prepos == self.ghost.pos and self.pacman.pos == self.ghost.prepos):
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
                
        return [self.st2ob(), rw, done]
    
    def wall_exists(self, coord1, coord2, walls):
        return [coord1, coord2] in walls or [coord2, coord1] in walls

    def add_walls(self):
        walls = []
        margin = 0.5
        walls.append([[margin, margin], [self.map_size_x + margin, margin],
                      [margin, margin], [margin, self.map_size_y + margin],
                      [margin, self.map_size_y + margin], [self.map_size_x + margin, self.map_size_y + margin],
                      [self.map_size_x + margin, margin], [self.map_size_x + margin, self.map_size_y + margin]])
        for ob_a in self.obstacles:
            for ob_b in self.obstacles:
                if self.wall_exists(ob_a, ob_b, walls):
                    continue
                if abs(ob_a[0] - ob_b[0]) == 1 and abs(ob_a[1] - ob_b[1]) == 0:
                    walls.append([ob_a, ob_b])
                elif abs(ob_a[1] - ob_b[1]) == 1 and abs(ob_a[0] - ob_b[0]) == 0:
                    walls.append([ob_b, ob_a])
        return walls

    @staticmethod
    def _faded_image(img, alpha):
        """Return a copy of img (H×W×3 or H×W×4) with opacity scaled by alpha."""
        arr = img.astype(float)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.shape[2] == 3:
            arr = np.concatenate([arr, np.ones((*arr.shape[:2], 1))], axis=2)
        result = arr.copy()
        result[:, :, 3] *= alpha
        return result

    @staticmethod
    def _rotate_sprite(img, direction):
        """
        Rotate a sprite so it faces the given direction.
        The base sprite is assumed to face east (right).
          'e' / None → no rotation
          'w'        → horizontal flip
          'n'        → 90° counter-clockwise (faces up)
          's'        → 90° clockwise (faces down)
        """
        if direction is None or direction == 'e':
            return img
        elif direction == 'w':
            return np.flip(img, axis=1)
        elif direction == 'n':
            return np.rot90(img, k=1)
        elif direction == 's':
            return np.rot90(img, k=3)
        return img

    def plot(self, trail=None, pacman_dir=None, trail_dirs=None):
        """Plot the current environment state.

        trail      : list of [pac_pos, ghost_pos] for past frames, oldest first.
                     Up to 2 entries; rendered as faded sprites.
        pacman_dir : direction the current Pacman sprite faces ('n','s','e','w').
                     Defaults to 'e' (right) if None.
        trail_dirs : list of directions matching each entry in trail.
                     Defaults to 'e' for all trail frames if None.
        """
        fig, ax = plt.subplots(figsize=(5, 5), edgecolor='black')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
        ax.set_aspect(1.0)
        for i in range(1, self.map_size_x + 1):
            ax.plot([1, self.map_size_x], [i, i], color='#1818A6', zorder=1, linewidth=2)
        for i in range(1, self.map_size_y + 1):
            ax.plot([i, i], [1, self.map_size_y], color='#1818A6', zorder=1, linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#000000')
        for wall in self.walls:
            x, y = zip(*wall)
            plt.plot(x, y, color='#000000', linewidth=10,
                     path_effects=[pe.Stroke(linewidth=15, foreground='#4663FF'), pe.Normal()], zorder=2)
        pellet_imagebox = OffsetImage(self.pellet_img, zoom=0.05)
        for pellet in self.pellets.remaining_pellets():
            a = AnnotationBbox(pellet_imagebox, (pellet[0], self.map_size_y + 1 - pellet[1]), frameon=False)
            ax.add_artist(a)
        # Faded trail sprites (oldest = most faded)
        if trail:
            trail_alphas = [0.2, 0.4]
            offset = 2 - len(trail)  # so the most-recent trail entry always gets alpha 0.4
            for i, (pp, gp) in enumerate(trail):
                alpha = trail_alphas[i + offset]
                t_dir = (trail_dirs[i] if trail_dirs and i < len(trail_dirs) else None)
                rotated_pac = self._rotate_sprite(self.pacman_img, t_dir)
                ax.add_artist(AnnotationBbox(
                    OffsetImage(self._faded_image(rotated_pac, alpha), zoom=0.025),
                    (pp[0], self.map_size_y + 1 - pp[1]), frameon=False))
                ax.add_artist(AnnotationBbox(
                    OffsetImage(self._faded_image(self.ghost_img, alpha), zoom=0.1),
                    (gp[0], self.map_size_y + 1 - gp[1]), frameon=False))
        # Current sprites (full opacity)
        rotated_pac = self._rotate_sprite(self.pacman_img, pacman_dir)
        pacman_imagebox = OffsetImage(rotated_pac, zoom=0.025)
        a = AnnotationBbox(pacman_imagebox,
                           (self.pacman.pos[0], self.map_size_y + 1 - self.pacman.pos[1]), frameon=False)
        ax.add_artist(a)
        ghost_imagebox = OffsetImage(self.ghost_img, zoom=0.1)
        b = AnnotationBbox(ghost_imagebox,
                           (self.ghost.pos[0], self.map_size_y + 1 - self.ghost.pos[1]), frameon=False)
        ax.add_artist(b)
        return fig, ax

    # generate display string
    def display(self):
        disp = self.map.copy()

        p_pos = self.pellets.remaining_pellets()
        for i in range(self.pellets.number_remaining_pellets()):
            disp[p_pos[i][1]] = self.replaceChar(disp[p_pos[i][1]], '*', p_pos[i][0])

        disp[self.pacman.pos[1]] = self.replaceChar(disp[self.pacman.pos[1]], 'P', self.pacman.pos[0])
        disp[self.ghost.pos[1]]   = self.replaceChar(disp[self.ghost.pos[1]],   'G', self.ghost.pos[0])
            
        return disp
    
    def replaceChar(self, st, c, idx):
        return st[0:idx] + c + st[idx+1:]
    
    # state to observation conversion
    def st2ob(self):
        gPosIdx = (self.ghost.pos[0]-1) + (self.ghost.pos[1]-1)*self.map_size_x
        gDirIdx = np.argmax( np.array(['n', 's', 'e', 'w']) == self.ghost.dir )
        pPosIdx = (self.pacman.pos[0]-1) + (self.pacman.pos[1]-1)*self.map_size_x
        peltIdx = np.sum( self.pellets.valid * 2**np.arange( len(self.pellets.valid) ) )
    
        return gPosIdx + gDirIdx*(self.map_size_x*self.map_size_y) + \
                         pPosIdx*(self.map_size_x*self.map_size_y) * 4 + \
                         peltIdx*(self.map_size_x*self.map_size_y) * 4 *(self.map_size_x*self.map_size_y)
        
        