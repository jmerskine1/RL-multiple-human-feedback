import numpy as np


""" --------------------------------------------------------------------------
Ghost - class
"""
class ghost:
    def __init__(self, x, y, map, behaviour='scatter'):
        self.init_pos = [x, y]
        self.reset()
        self.moves = ['n','s','e','w']

        self.dir = 's'
        self.map = map
        
        self.P_change_direction = 0.5
        self.behaviour = behaviour        
        return
    
    def reset(self, x=[], y=[]):
        if x==[]:
            self.pos = self.init_pos
        else:
            self.pos = [x,y]
            
        self.prepos = self.pos       
        return
    
    def move(self,pacman_pos):

        available_moves = self.moves.copy()
        cantMove=True
        
        while cantMove:
            n = len(available_moves)

            if n == 0:
                raise ValueError("No available moves to choose from.")

            if self.behaviour == 'chase':
                self.dir = self.chase(pacman_pos,available_moves)
            
            elif self.behaviour == 'scatter':
                probs = []
                if self.dir in available_moves:
                    probs = []
                    for move in available_moves:
                        if move == self.dir:
                            probs.append(0.5)
                        else:
                            probs.append(0.5 / (n - 1))
                else:
                    # Split evenly among all available directions
                    probs = [1.0 / n] * n
                
                self.dir = np.random.choice(available_moves, p=probs)


            #check if the same movement is possible
            newPos = self.newPos(self.pos, self.dir)

            if newPos == self.pos:
                available_moves.remove(self.dir) #type:ignore

            else:
                # actually move
                self.prepos = self.pos
                self.pos = newPos
                cantMove = False
                
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
    
    def chase(self,objective,moves):
        # calculate euclidean distance between ghost and pacman for each available move
        distances = []
        for move in moves:
            new_pos = self.newPos(self.pos,move)
            x = new_pos[0] - objective[0]
            y = new_pos[1] - objective[1]
            distance = np.sqrt(x**2 + y**2)
            if new_pos == self.prepos:
                distance += 1000 # high penalty for going backwards ensures "last-resort"
            distances.append(distance)

        
        return moves[distances.index(min(distances))]