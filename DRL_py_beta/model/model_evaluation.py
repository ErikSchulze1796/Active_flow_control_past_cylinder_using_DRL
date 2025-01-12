import torch as pt
from sklearn.metrics import r2_score


def evaluate_model(model: pt.nn.Module,
                    features_norm: pt.Tensor,
                    labels_norm: pt.Tensor,
                    model_path: str,
                    n_steps_history: int,
                    every_nth_element: int,
                    n_inputs: int):
    """This function evaluates the given model for given features and labels

    Parameters
    ----------
    model : pt.nn.Module
        Model to be evaluated
    features_norm : pt.Tensor
        Features the model should predict from
    labels_norm : pt.Tensor
        Labels data the prediction is tested against
    model_path : str
        Path where the model is stored
    n_steps_history : int
        Number of subsequent states to be included in input
    every_nth_element : int
        Every nth pressure sensor to be kept as input
    n_inputs : int
        Number of input features for one training example of the model

    Returns
    -------
    list, list, list, torch.Tensor, torch.Tensor, torch.Tensor
        test_loss_l2: Loss calculated using the L2 norm
        test_loss_lmax: Loss calculated using max norm
        r2score: R² (R squared) score of the prediction
        prediction_p: Predicted pressure values for all 400 sensors along the cylinder's surface
        prediction_cd: Predicted drag coefficient
        prediction_cl: Predicted lift coefficient
    """
    model.eval()
    test_loss_l2 = pt.zeros(labels_norm.shape[0])
    test_loss_lmax = pt.zeros(labels_norm.shape[0])
    r2score = pt.zeros(labels_norm.shape[0])
    prediction_p = pt.zeros(labels_norm[:,:-2].shape)
    prediction_cd = pt.zeros(labels_norm[:,-2].shape)
    prediction_cl = pt.zeros(labels_norm[:,-1].shape)
    # model.load_state_dict((pt.load(model_path)))
    full_pred_norm = pt.zeros(labels_norm.shape)
    for idx_t, state_t in enumerate(features_norm):

        pred_norm = model(state_t).squeeze().detach()
        full_pred_norm[idx_t,:] = pred_norm
        
        prediction_p[idx_t] = pred_norm[:-2]
        prediction_cd[idx_t] = pred_norm[-2]
        prediction_cl[idx_t] = pred_norm[-1]

        test_loss_l2[idx_t] = (pred_norm - labels_norm[idx_t,:]).square().mean()
        test_loss_lmax[idx_t] = (pred_norm - labels_norm[idx_t,:]).absolute().max() / 2
        r2score[idx_t] = r2_score(labels_norm[idx_t,:], pred_norm)
        
        print("\r", "L2 loss: {:1.4f}, Lmax loss: {:1.4f}, r2 score: {:1.4f}".format(test_loss_l2[idx_t], test_loss_lmax[idx_t], r2score[idx_t]), end="\r")

        if idx_t == labels_norm.shape[0]-1:
            break
        features_norm[idx_t+1,(n_steps_history-1)*n_inputs:-1] = pred_norm[:-2][::every_nth_element]
    
    return test_loss_l2, test_loss_lmax, r2score, prediction_p, prediction_cd, prediction_cl

def evaluate_model_new(model: pt.nn.Module,
                    features: pt.Tensor,
                    labels: pt.Tensor,
                    model_path: str,
                    n_steps_history: int,
                    every_nth_element: int,
                    n_inputs: int):
    """This function evaluates the given model for given features and labels

    Parameters
    ----------
    model : pt.nn.Module
        Model to be evaluated
    features_norm : pt.Tensor
        Features the model should predict from
    labels_norm : pt.Tensor
        Labels data the prediction is tested against
    model_path : str
        Path where the model is stored
    n_steps_history : int
        Number of subsequent states to be included in input
    every_nth_element : int
        Every nth pressure sensor to be kept as input
    n_inputs : int
        Number of input features for one training example of the model

    Returns
    -------
    list, list, list, torch.Tensor, torch.Tensor, torch.Tensor
        test_loss_l2: Loss calculated using the L2 norm
        test_loss_lmax: Loss calculated using max norm
        r2score: R² (R squared) score of the prediction
        prediction_p: Predicted pressure values for all 400 sensors along the cylinder's surface
        prediction_cd: Predicted drag coefficient
        prediction_cl: Predicted lift coefficient
    """
    model.eval()
    test_loss_l2 = pt.zeros(labels.shape[0])
    test_loss_lmax = pt.zeros(labels.shape[0])
    r2score = pt.zeros(labels.shape[0])
    prediction_p = pt.zeros(labels[:,:-2].shape)
    prediction_cd = pt.zeros(labels[:,-2].shape)
    prediction_cl = pt.zeros(labels[:,-1].shape)
    features_predict = pt.zeros(features.shape)
    features_predict[0,:] = features[0,:]

    full_pred_norm = pt.zeros(labels.shape)
    for idx_t, state_t in enumerate(features_predict):
        previous_states = state_t[n_inputs:].clone()

        pred = model(state_t).squeeze().detach()
        full_pred_norm[idx_t,:] = pred
        
        prediction_p[idx_t] = pred[:-2]
        prediction_cd[idx_t] = pred[-2]
        prediction_cl[idx_t] = pred[-1]
        # we normalize the maximum error with the range of the scaled/normalized data,
        # which is 1-(-1)=2
        # test_loss_l2.append((pred_norm - labels_norm[idx_t,:]).square().mean())
        # test_loss_lmax.append((pred_norm - labels_norm[idx_t,:]).absolute().max() / 2)
        # r2score.append(r2_score(labels_norm[idx_t,:], pred_norm))
        
        test_loss_l2[idx_t] = (pred - labels[idx_t,:]).square().mean()
        test_loss_lmax[idx_t] = (pred - labels[idx_t,:]).absolute().max() / 2
        r2score[idx_t] = r2_score(labels[idx_t,:], pred)
        
        print("\r", "L2 loss: {:1.4f}, Lmax loss: {:1.4f}, r2 score: {:1.4f}".format(test_loss_l2[idx_t], test_loss_lmax[idx_t], r2score[idx_t]), end="\r")

        if idx_t == labels.shape[0]-1:
            break
        
        next_action = features[idx_t+1,-1]
        next_state = pt.cat((pred[:-2][::every_nth_element], next_action.unsqueeze(dim=0)), dim=0)
        next_combined_feature = pt.cat((previous_states, next_state),dim=0)
        features_predict[idx_t+1,:] = next_combined_feature
    
    return test_loss_l2, test_loss_lmax, r2score, prediction_p, prediction_cd, prediction_cl

def get_predictions(model: pt.nn.Module,
                    features: pt.Tensor,
                    labels: pt.Tensor,
                    model_path: str,
                    n_steps_history: int,
                    every_nth_element: int,
                    n_inputs: int):
    """This function evaluates the given model for given features and labels

    Parameters
    ----------
    model : pt.nn.Module
        Model to be evaluated
    features_norm : pt.Tensor
        Features the model should predict from
    labels_norm : pt.Tensor
        Labels data the prediction is tested against
    model_path : str
        Path where the model is stored
    n_steps_history : int
        Number of subsequent states to be included in input
    every_nth_element : int
        Every nth pressure sensor to be kept as input
    n_inputs : int
        Number of input features for one training example of the model

    Returns
    -------
    list, list, list, torch.Tensor, torch.Tensor, torch.Tensor
        test_loss_l2: Loss calculated using the L2 norm
        test_loss_lmax: Loss calculated using max norm
        r2score: R² (R squared) score of the prediction
        prediction_p: Predicted pressure values for all 400 sensors along the cylinder's surface
        prediction_cd: Predicted drag coefficient
        prediction_cl: Predicted lift coefficient
    """
    model.eval()
    test_loss_l2 = pt.zeros(labels.shape[0])
    test_loss_lmax = pt.zeros(labels.shape[0])
    r2score = pt.zeros(labels.shape[0])
    prediction_p = pt.zeros(labels[:,:-2].shape)
    prediction_cd = pt.zeros(labels[:,-2].shape)
    prediction_cl = pt.zeros(labels[:,-1].shape)
    features_predict = pt.zeros(features.shape)
    features_predict[0,:] = features[0,:]

    full_pred_norm = pt.zeros(labels.shape)
    for idx_t, state_t in enumerate(features_predict):
        previous_states = state_t[n_inputs:].clone()

        pred = model(state_t).squeeze().detach()
        full_pred_norm[idx_t,:] = pred
        
        prediction_p[idx_t] = pred[:-2]
        prediction_cd[idx_t] = pred[-2]
        prediction_cl[idx_t] = pred[-1]
        # we normalize the maximum error with the range of the scaled/normalized data,
        # which is 1-(-1)=2
        # test_loss_l2.append((pred_norm - labels_norm[idx_t,:]).square().mean())
        # test_loss_lmax.append((pred_norm - labels_norm[idx_t,:]).absolute().max() / 2)
        # r2score.append(r2_score(labels_norm[idx_t,:], pred_norm))
        
        test_loss_l2[idx_t] = (pred - labels[idx_t,:]).square().mean()
        test_loss_lmax[idx_t] = (pred - labels[idx_t,:]).absolute().max() / 2
        r2score[idx_t] = r2_score(labels[idx_t,:], pred)
        
        # print("\r", "L2 loss: {:1.4f}, Lmax loss: {:1.4f}, r2 score: {:1.4f}".format(test_loss_l2[idx_t], test_loss_lmax[idx_t], r2score[idx_t]), end="\r")

        if idx_t == labels.shape[0]-1:
            break
        
        next_action = features[idx_t+1,-1]
        next_state = pt.cat((pred[:-2][::every_nth_element], next_action.unsqueeze(dim=0)), dim=0)
        next_combined_feature = pt.cat((previous_states, next_state),dim=0)
        features_predict[idx_t+1,:] = next_combined_feature
    
    return full_pred_norm, prediction_p, prediction_cd, prediction_cl
