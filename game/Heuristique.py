import numpy as np

path = "param_heuristique.txt"

params = dict()
params['seuil_haut_cote'] = 0.5
params['seuil_bas_cote'] = 0.4
params['seuil_haut_ttdroit'] = 2.5
params['seuil_bas_ttdroit'] = 1.25

#tourner en fonction des seuils
params['action_tourner_seuil_haut_cote'] = 0.4
params['action_tourner_seuil_milieu_cote'] = 1.
params['action_tourner_seuil_bas_cote'] = 1.5

#accélération en fonction de la distance de devant
params['action_accel_seuil_haut_ttdroit'] = -0.5
params['action_accel_seuil_milieu_ttdroit'] = 0.33
params['action_accel_seuil_bas_ttdroit'] = 0.5

params['action_accel_seuil_haut_cote'] = -0.5
params['action_accel_seuil_milieu_cote'] = 0.
params['action_accel_seuil_bas_cote'] = 0.5


def get_aleat_param(params):
    params = dict()
    params['seuil_haut_cote'] = 0.5 #aleatoire entre 0.4 et 0.6
    params['seuil_bas_cote'] = 0.4 #aleatoireentre 0.2 et seuil haut
    params['seuil_haut_ttdroit'] = 2.5 #aleatoire entre 2 et 3
    params['seuil_bas_ttdroit'] = 1.25 #aleatoire entre seuil haut et 0.9
    
    #tourner en fonction des seuils
    params['action_tourner_seuil_haut_cote'] = 0.4 #aleatoire entre 0 et 0.9 
    params['action_tourner_seuil_milieu_cote'] = 1. #aleatoire entre 0.9 et 1.5
    params['action_tourner_seuil_bas_cote'] = 1.5 #aleatoire entre seuil milieu et 2
    
    #accélération en fonction de la distance de devant
    params['action_accel_seuil_haut_ttdroit'] = -0.5 #aleatoire entre -1 et -0.1
    params['action_accel_seuil_milieu_ttdroit'] = 0.33 #aleatoire entre 0 et 0.9
    params['action_accel_seuil_bas_ttdroit'] = 0.5 #aleatoire entre seuil milieu et 1
    
    params['action_accel_seuil_haut_cote'] = -0.5
    params['action_accel_seuil_milieu_cote'] = 0.
    params['action_accel_seuil_bas_cote'] = 0.5
    return 

# =============================================================================
# def ecrit_params(path, params):
#     f = open(path, "w")
#     f.write(str())
#     f.write(" ")
# =============================================================================

def heuristic(agent_input, params= params):
      
    capteurs = agent_input[5:12].copy()
    direction = np.argmax(capteurs)
    
    capteurs_ = [-1*x for x in capteurs]
    argtriee = np.argsort(capteurs_)
    print(argtriee)
    
    if (argtriee[6] == 0 and argtriee[4]==1):
        return tout_droit(capteurs, params)
    
    elif  ((argtriee[4] == 0 and argtriee[5]==1 and argtriee[6]==2) and (argtriee[2] == 3 or argtriee[3]==3)) :
        return tout_droit(capteurs, params)
    
    elif (argtriee[6] == 0 and argtriee[3]== 1) or (argtriee[3] == 0 and argtriee[6]== 1):
        return tout_droit(capteurs, params)
    
    elif  ((argtriee[6] == 0 and argtriee[5]==1) or (argtriee[5] == 0 and argtriee[6]==1)) and ((argtriee[3]==2 and argtriee[4]==3) or (argtriee[3]==3 and argtriee[4]==2)):
        return tout_droit(capteurs, params)
    
    #pour le dernier virage
    
    else :
        print("direction :", direction)
        if direction == 0 or direction ==1:
            print("on vas a gauche !")
            return gauche_toute(capteurs, params)
        elif direction == 2:
            print("on vas a gauche !")
            return gauche(capteurs, params)
        elif direction == 3:
            print("on vas tout droit")
            return tout_droit(capteurs, params)
        elif direction == 4:
            print("on vas a droite !")
            return droite(capteurs, params)
        elif direction == 6 or direction ==5:
            print("on vas a droite !")
            return droite_toute(capteurs, params)


def gauche_toute(agent_input, param):
    """
    vas a gauche toute, mais regle son
    accélération en fonction du capteur de devant
    """
    action = [0.,0.,0.,0.,0.]
    if agent_input[3] < param['seuil_bas_ttdroit']:
        action[0] += param['action_tourner_seuil_bas_cote'] #tourner
        action[2] += param['action_accel_seuil_bas_cote'] #accelerer
        
    elif param['seuil_bas_ttdroit'] <= agent_input[3] <= param['seuil_haut_ttdroit']:
        action[0] += param['action_tourner_seuil_bas_cote'] #tourner
        action[2] += param['action_accel_seuil_milieu_cote'] #accelerer
        
    elif param['seuil_haut_ttdroit'] < agent_input[3] :
        action[0] += param['action_tourner_seuil_bas_cote']   
        action[2] += param['action_accel_seuil_haut_cote'] #accelerer
    return action


def gauche(agent_input, param):
    """
    accelere plus ou moins en fonction de la distance de devant
    tourne plus ou moins en fonction de la distance du côté
    """
    action = [0.,0.,0.,0.,0.]
    if agent_input[0] < param['seuil_bas_cote']:
        action[0] += param['action_tourner_seuil_bas_cote'] #tourner
    elif param['seuil_bas_cote'] < agent_input[0] < param['seuil_haut_cote'] :
        action[0] += param['action_tourner_seuil_milieu_cote'] #tourner
    else :
        action[0] += param['action_tourner_seuil_haut_cote'] #tourner
        
    if agent_input[3] < param['seuil_bas_ttdroit']:
        action[2] += param['action_accel_seuil_bas_cote'] #accelerer
        
    elif param['seuil_bas_ttdroit'] <= agent_input[3] <= param['seuil_haut_ttdroit']:
        action[2] += param['action_accel_seuil_milieu_cote'] #accelerer
        
    elif param['seuil_haut_ttdroit'] < agent_input[3] :
        action[2] += param['action_accel_seuil_haut_cote'] #accelerer
    return action


def tout_droit(agent_input, param):
    action = [0.,0.,0.,0.,0.]  
    if agent_input[3] < param['seuil_bas_ttdroit'] :
        action[2] += param['action_accel_seuil_haut_ttdroit']
    elif param['seuil_bas_ttdroit'] < agent_input[3] < param['seuil_haut_ttdroit']:
        action[2] += param['action_accel_seuil_milieu_ttdroit']
    else:
        action[2] += param['action_accel_seuil_bas_ttdroit']
    return action


def on_fonce_tout_droit(agent_input, param):
    action = [0.,0.,0.,0.,0.]  
    action[2] += param['action_accel_seuil_milieu_ttdroit']
    return action


def droite(agent_input, param):
    """
    accelere plus ou moins en fonction de la distance de devant
    tourne plus ou moins en fonction de la distance du côté
    """
    action = [0.,0.,0.,0.,0.]
    if agent_input[0] < param['seuil_bas_cote']:
        action[1] += param['action_tourner_seuil_bas_cote'] #tourner
    elif param['seuil_bas_cote'] < agent_input[0] < param['seuil_haut_cote'] :
        action[1] += param['action_tourner_seuil_milieu_cote'] #tourner
    else :
        action[1] += param['action_tourner_seuil_haut_cote'] #tourner
        
    if agent_input[3] < param['seuil_bas_ttdroit']:
        action[2] += param['action_accel_seuil_bas_cote'] #accelerer
        
    elif param['seuil_bas_ttdroit'] <= agent_input[3] <= param['seuil_haut_ttdroit']:
        action[2] += param['action_accel_seuil_milieu_cote'] #accelerer
        
    elif param['seuil_haut_ttdroit'] < agent_input[3] :
        action[2] += param['action_accel_seuil_haut_cote'] #accelerer
    return action


def droite_toute(agent_input, param):
    """
    vas a droite toute, mais regle son
    accélération en fonction du capteur de devant
    """
    action = [0.,0.,0.,0.,0.]
    if agent_input[3] < param['seuil_bas_ttdroit']:
        action[1] += param['action_tourner_seuil_bas_cote'] #tourner
        action[2] += param['action_accel_seuil_bas_cote'] #accelerer
        
    elif param['seuil_bas_ttdroit'] <= agent_input[3] <= param['seuil_haut_ttdroit']:
        action[1] += param['action_tourner_seuil_bas_cote'] #tourner
        action[2] += param['action_accel_seuil_milieu_cote'] #accelerer
        
    elif param['seuil_haut_ttdroit'] < agent_input[3] :
        action[1] += param['action_tourner_seuil_bas_cote']   
        action[2] += param['action_accel_seuil_haut_cote'] #accelerer
    return action

    
    