import pygame
import gym
from officesimenv import OfficeEnv


# font = pygame.font.SysFont(None, 20)
env = OfficeEnv()
isRunning = True
move_flag = True
# step = 0
_ = env.reset()
# env.render()
while isRunning:
        # env.clock_obj.tick(env.FPS)
        env.render()
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                isRunning = False
                

            if evt.type == pygame.KEYDOWN:
                if evt.key == pygame.K_SPACE:
                    move_flag = not(move_flag)

            if evt.type == pygame.MOUSEMOTION:
                mx , my = pygame.mouse.get_pos()
                caption = f"({mx},{my})"
                pygame.display.set_caption(caption)
                # print("mx,my : ",mx,my)
              
              
        fa =[]      
        if move_flag:
             # step +=1
            
             env.step([None,None])

     
print("num steps",env.simulation_step)

env.close()       
