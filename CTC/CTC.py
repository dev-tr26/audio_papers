import torch 

# smallest  float32 number possible 
NEG_INF = torch.finfo(torch.float32).min

def CTC_LOSS(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='none'):
    #  log_probs: (T, B, C) - log probabilities (log softmax output from model)
    #  targets: (B, S) - target sequences
    #  input_lengths: (B,) 
    #  target_lengths: (B,) 
    #  blank: int - index of the blank label
    seq_len, batch_size, num_classes = log_probs.shape 
    B = torch.arange(batch_size)

    # targets : [28, 1, 17, 21, 0] -> append a blank token at end 
    # its just for indexing not used anywhere in inference nor include it in loss 
    _t_a_r_g_e_t_s = torch.cat([targets, torch.zeros(batch_size, 1, device=log_probs.device, dtype=torch.long)], dim=-1)
    print(_t_a_r_g_e_t_s)
    
    # _t_a_r_g_e_t_s : [0, 28, 0, 1, 0, 17, 0, 21, 0, 0] -> insert blank tokens b/w targets 
    _t_a_r_g_e_t_s = torch.stack([torch.full_like(_t_a_r_g_e_t_s, blank), _t_a_r_g_e_t_s], dim=-1).flatten(start_dim=-2)
    print(_t_a_r_g_e_t_s)

    # rule 1 : if two same consecutive targets -> use blank in b/w
    # rule 2:  stay o same letter or move1 letter or two letters forward 
     
     
    # flag bool if two consecutive labels same or different 
    # diff_labels = _t_a_r_g_e_t_s[:, :-2] != _t_a_r_g_e_t_s[:,2:]
    print(_t_a_r_g_e_t_s.shape)

    # prepending [false,false] to keep the shaoe same as _t_a_r_g_e_t_s  
    diff_labels = torch.cat([torch.tensor([[False, False]], device=log_probs.device).expand(batch_size,-1), (_t_a_r_g_e_t_s[:,:-2] != _t_a_r_g_e_t_s[:,2:])], dim=-1)
    print(diff_labels.shape)
    print(diff_labels)

    # we gather probs of this indexes from log probs at every timestep along our logits 
    # p(0 |T=0) , p(28|t=0) , p(0|t=0), p(char or blank token / at time 0) 
    # p(0 |T=1) , p(28 |t=1) , p(char or blank token / time 1)
    # ......
    # .....
    # p(0| T=t) 
    
    # gather log probs 
    print(log_probs.shape)
    print(_t_a_r_g_e_t_s)
    
    # we need all the probs at time 0 and time 1 so we first need to create a index matrix i.e of targets 
    
    index  = _t_a_r_g_e_t_s.expand(seq_len, -1, -1)
    print(index)
    
    # gather index values 0, 1, 2,3 etc from the rows of index matrix 
    log_probs_ = log_probs.gather(dim=-1, index=index)
    print(log_probs_.shape)

    # seq_len, target_len log alpha matrix  with -ve infinity so we adoing things in log probs space log 0 = -ve infinity 
    # log_alpha = torch.full((seq_len,batch_size,_t_a_r_g_e_t_s.shape[-1]), NEG_INF).to(device)
    # print(log_alpha.shape)
    
    # check 1 or 2 step back , so correct indexing 
    log_alpha = torch.full((seq_len, batch_size,2 + _t_a_r_g_e_t_s.shape[-1]), NEG_INF).to(device)
    print(log_alpha.shape)
    
    log_alpha[0, :,2] = log_probs[0, :, blank] 
    print(log_probs.shape)

    # first timestamp  for blank and c token 
    # log_alpha[0, :, 2+1]
    print(_t_a_r_g_e_t_s)
    print(_t_a_r_g_e_t_s[:,1]) 
    # in logprobs indexed at  1, 16 of getting 16 at first timestep 
    print(log_probs[_t_a_r_g_e_t_s[:,1]])
    
    # log probs for batch size 2  all samples in batch 
    print(log_probs)
    
    # log prob of i for 0th sample at 0th timestep 
    print(log_probs[0,B, _t_a_r_g_e_t_s[:,1]]) # this is log prob of 1st actual target for the 1st sample in the batch 
    
    print(B)
    print(_t_a_r_g_e_t_s[:,1])
    print(log_probs.shape)
    log_alpha[0, :, 2+1] = log_probs[0, B, _t_a_r_g_e_t_s[:,1]]
    
    for T_ in range(1, T):
        log_probs_T_ = log_probs_[T_] # all probs of current timestamp
        
        log_alpha_T_prev_state = log_alpha[T_-1, : , 2:]   # all probs of keeping in same timestamp going forward 
        log_alpha_T_prev_next_state =  log_alpha[T_-1, :, 1:-1] # all probs of moving one step forward in time 
        
        # [pad, pad , T, T, T, T, T]  2 to n all probs  == T, T, T, T, T
        # 1 to -1 all probs going 1 step backward ( 1 step transtion )== pad , T, T, T, T
        # 0 to -2 all probs going 2 step backward ( 2 step transition ) == pad, pad, T, T, T
        
        log_alpha_two_step_transition = torch.where(diff_labels, input=log_alpha[T_-1, :, :-2], other = NEG_INF) # mask probs true for only valid transition check 
        # valid transition diff_labels = true , invalid transition probs = 0 so -ve infinity 
        
        prob = torch.logsumexp(torch.stack([log_alpha_T_prev_next_state, log_alpha_T_prev_state, log_alpha_two_step_transition ]), dim=0)
        
        print(log_probs_T_.shape)
        print(prob.shape)
        log_alpha[T_, :, 2:] = log_probs_T_ + prob 
        
    final_log_alpha = log_alpha[input_lengths-1, B]
    print(final_log_alpha)

    # [A, B, C, D, E, F, G, H, I, J]
    # after padding [- , A, -, B, -, C, -, D, -, E, -, F, -, G, -, H, -, I, -, J, -, -]
    # we can end with either J or blank token just after it 
    ending_on_blank_index =  2 + target_lengths*2 
    ending_on_label_index =  2 + target_lengths*2 - 1 
    
    indexes_to_get = torch.stack([ending_on_label_index, ending_on_blank_index], dim=-1)
    print(indexes_to_get)
    label_or_blank_ending_log_alphas = final_log_alpha.gather(dim=-1, index=indexes_to_get)
    print(label_or_blank_ending_log_alphas)


    # we want to maximize prob but nn minimize loss so we minimize negative of probs 
    ctc_loss = - torch.logsumexp(label_or_blank_ending_log_alphas, dim=-1)
    
    
    if reduction == "none":
        return ctc_loss
    elif reduction == "sum":
        return torch.sum(ctc_loss)
    elif reduction == "mean":
        return torch.mean(ctc_loss)
     





if __name__ == "__main__":
    # 128 features extracted from convolution, batch_size, no of possibilities for different chars  
    T,B,C = 128, 2, 32
    t =50 # target char sequence length
    blank = 0   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    atol = 1e-3
    
    logits = torch.randn(T,B,C).requires_grad_().to(device)
    
    # any random num b/w 1 and 31 as target seq is 32 
    targets = torch.randint(blank+1, C, (B,t), dtype=torch.long).to(device)
    
    # vector of shape batch size tells loss fun what is length of each input seq
    # as we have no padding 
    input_lengths = torch.full((B,), T, dtype=torch.long).to(device)   # i/p seq len 
    target_lengths = torch.full((B,), t, dtype=torch.long).to(device)  # target seq len 
    
    log_probs = logits.log_softmax(dim=-1).to(device)
    
    my_ctc_loss = CTC_LOSS(log_probs, targets, input_lengths, target_lengths)
    my_loss_grads = torch.autograd.grad(my_ctc_loss.mean(), logits, retain_graph=True)[0]

    torch_ctc_loss = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths,target_lengths, blank=0, reduction='none')
    torch_loss_grad = torch.autograd.grad(torch_ctc_loss.mean(), logits, retain_graph=True)[0]
    
    print(f"CTC Loss Matches:", torch.allclose(torch_ctc_loss, my_ctc_loss, atol=atol))
    print(f"CTC Gradients Match:", torch.allclose(torch_loss_grad, my_loss_grads, atol=atol))