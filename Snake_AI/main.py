import pygame
import math
import random
from snake import Snake
import neat

SIZE=40

#init class
class Snake:
    def __init__(self):
        self.pos=[(120,120),(80,120),(40,120)]
        self.head=pygame.image.load("snake_rs/block.png")
        self.head =pygame.transform.scale(self.head, (40, 40))
        self.image=pygame.image.load("snake_rs/player1.png")
        self.image =pygame.transform.scale(self.image, (40, 40))
        self.dir='down'
        self.applenum=0
        self.count=10
        self.is_alive = True

        self.applex=random.randrange(1,19)*SIZE
        self.appley=random.randrange(1,19)*SIZE
        self.appleimage=pygame.image.load("snake_rs/apple.png")
        self.appleimage =pygame.transform.scale(self.appleimage, (40, 40))
    
    def move(self): #move snake
        if self.dir== 'right':
            self.pos.insert(1,self.pos[0])
            self.pos.pop()
            self.pos[0]=tuple(map(sum, zip(self.pos[0], (SIZE,0))))
        if self.dir== 'left':
            self.pos.insert(1,self.pos[0])
            self.pos.pop()
            self.pos[0]=tuple(map(sum, zip(self.pos[0], (-SIZE,0))))
        if self.dir== 'up':
            self.pos.insert(1,self.pos[0])
            self.pos.pop()
            self.pos[0]=tuple(map(sum, zip(self.pos[0], (0,-SIZE))))
        if  self.dir== 'down':
            self.pos.insert(1,self.pos[0])
            self.pos.pop()
            self.pos[0]=tuple(map(sum, zip(self.pos[0], (0,SIZE))))

    def vision(self): #take input data
        dir=[0,0,0,0]
        #calculate angle between the snake and apple
        if self.appley-self.pos[0][1] !=0:
            angle = (self.applex-self.pos[0][0])/(self.appley-self.pos[0][1])
        elif self.applex-self.pos[0][0] >0:
            angle = 1000000000000000000000 
        elif self.applex-self.pos[0][0] <=0:
            angle = -1000000000000000000000

        dir[0]=angle

        #take direction of the snake     
        if self.dir=='right':
            dir[2]=1
        if self.dir=='left':
            dir[2]=0
        if self.dir=='up':
            dir[1]=0
        if self.dir=='down':
            dir[1]=1

        #calculate the distance between apple and snake     
        pos = math.sqrt((self.applex-self.pos[0][0])**2+(self.appley-self.pos[0][1])**2)
        dir[3]=pos
        return tuple(dir)
    
    def draw(self,screen):
        self.length=len(self.pos)
        for i in range(1,self.length):
            screen.blit(self.image,(self.pos[i]))
        screen.blit(self.head,self.pos[0])
        screen.blit(self.appleimage,(self.applex,self.appley))
    
    def play(self):
        self.move()
        #check for collision
        if collision(self.applex,self.appley,self.pos[0]):
            #self.snake.pos.append((-100,-100))
            self.applenum+=1
            self.applex= random.randrange(1,19)*SIZE
            self.appley= random.randrange(1,19)*SIZE
            self.count=10
        
        self.count-=0.1
        if self.count<0:
            self.is_alive=False
        
        #check offscreen
        if self.pos[0][0] < 0:
            lst = list(self.pos[0])
            lst[0] = 800
            self.pos[0]=tuple(lst)
        if self.pos[0][0] > 800:
            lst = list(self.pos[0])
            lst[0] = 0
            self.pos[0]=tuple(lst)
        if self.pos[0][1] < 0:
            lst = list(self.pos[0])
            lst[1] = 800
            self.pos[0]=tuple(lst)
        if self.pos[0][1] >800:
            lst = list(self.pos[0])
            lst[1] = 0
            self.pos[0]=tuple(lst)  
    
    def control(self, up=None, right=None): #change snake direction
        if right == True and self.dir != 'left':
            self.dir='right'

        if right == False and self.dir != 'right':
            self.dir='left'

        if up == True and self.dir != 'down':
            self.dir='up'

        if up == False and self.dir != 'up':
            self.dir='down'

    def get_alive(self):
        return self.is_alive


def collision(x1, y1,pos):
    x2,y2=pos
    if x1<=x2 and x2<x1+SIZE:
        if y1 <=y2 and y2 < y1+SIZE:
            return True
    return False


def run_snake(genomes, config): #main loop
        pygame.init()
        nets = []
        snakes = []
        clock = pygame.time.Clock()

        for id, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)
            nets.append(net)
            g.fitness = 0
            snakes.append(Snake())    
            
        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
            screen = pygame.display.set_mode((800, 800))
            screen.fill((255,255,255))
            # Input my data and get result from network
            for index, snake in enumerate(snakes):
                output = nets[index].activate(snake.vision())    
                decision = output.index(max(output))
                if decision == 0:
                    snake.control(right=True)
                elif decision == 1:
                    snake.control(right=False)
                elif decision == 2:
                    snake.control(up=True)
                elif decision == 3: 
                    snake.control(up=False)

            #Check     
            remain_snake = 0
            for i,snake in enumerate(snakes):
                if snake.get_alive():
                    remain_snake += 1
            if remain_snake == 0:
                break
            for snake in snakes:
                if snake.get_alive():
                    snake.draw(screen)
                    snake.play()
            
            #update fitness
            for index, snake in enumerate(snakes):
                genomes[index][1].fitness-=0.1
                genomes[index][1].fitness += snake.applenum
                    
            pygame.display.flip()
            clock.tick(20)

def run_neat(config):
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.run(run_snake, 1000)

if __name__ == "__main__":
    config_path = "./config2.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    run_neat(config)