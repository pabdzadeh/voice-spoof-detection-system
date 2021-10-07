import sys
import numpy as np
import evaluation_metrics as em
import matplotlib.pyplot as plt

def evaluate_tDCF_asvspoof19(cm_score_file, asv_score_file, legacy):

    # Fix tandem detection cost function (t-DCF) parameters
    if legacy:
        Pspoof = 0.05
        cost_model = {
            'Pspoof': Pspoof,  # Prior probability of a spoofing attack
            'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
            'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
            'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
            'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
            'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
            'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
        }
    else:
        Pspoof = 0.05
        cost_model = {
            'Pspoof': Pspoof,  # Prior probability of a spoofing attack
            'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
            'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
            'Cmiss': 1,  # Cost of tandem system falsely rejecting target speaker
            'Cfa': 10,  # Cost of tandem system falsely accepting nontarget speaker
            'Cfa_spoof': 10,  # Cost of tandem system falsely accepting spoof
        }

    # Load organizers' ASV scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(np.float)

    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_keys = cm_data[:, 1]
    cm_scores = cm_data[:, 2].astype(np.float)

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scores[asv_keys == 'target']
    non_asv = asv_scores[asv_keys == 'nontarget']
    spoof_asv = asv_scores[asv_keys == 'spoof']

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']

    # EERs of the standalone systems and fix ASV operating point to EER threshold
    eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]


    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)


    # Compute t-DCF
    if legacy:
        tDCF_curve, CM_thresholds = em.compute_tDCF_legacy(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, True)
    else:
        tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, True)

    # Minimum t-DCF
    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]
    min_tDCF_threshold = CM_thresholds[min_tDCF_index];

    # compute DET of CM and get Pmiss and Pfa for the selected threshold t_CM
    Pmiss_cm, Pfa_cm, CM_thresholds = em.compute_det_curve(bona_cm, spoof_cm)
    Pmiss_t_CM = Pmiss_cm[CM_thresholds == min_tDCF_threshold]
    Pfa_t_CM = Pfa_cm[CM_thresholds == min_tDCF_threshold]


    print('ASV SYSTEM')
    print('   EER            = {:8.5f} % (Equal error rate (target vs. nontarget discrimination)'.format(eer_asv * 100))
    print('   Pfa            = {:8.5f} % (False acceptance rate of nontargets)'.format(Pfa_asv * 100))
    print('   Pmiss          = {:8.5f} % (False rejection rate of targets)'.format(Pmiss_asv * 100))
    if legacy:
        print('   1-Pmiss,spoof  = {:8.5f} % (Spoof false acceptance rate)'.format((1 - Pmiss_spoof_asv) * 100))
    else:
        print('   Pfa,spoof  = {:8.5f} % (Spoof false acceptance rate)'.format((1 - Pmiss_spoof_asv) * 100))

    print('\nCM SYSTEM')
    print('   EER                  = {:8.5f} % (Equal error rate for countermeasure)'.format(eer_cm * 100))
    print('   Pfa(t_CM_min_tDCF)   = {:8.5f} % (False acceptance rate of spoofs)'.format(Pfa_t_CM[0] * 100))
    print('   Pmiss(t_CM_min_tDCF) = {:8.5f} % (Miss (false rejection) rate of bonafide)'.format(Pmiss_t_CM[0] * 100))

    print('\nTANDEM')
    print('   min-tDCF       = {:8.5f}'.format(min_tDCF))


    # Visualize ASV scores and CM scores
    plt.figure()
    ax = plt.subplot(121)
    plt.hist(tar_asv, histtype='step', density=True, bins=50, label='Target')
    plt.hist(non_asv, histtype='step', density=True, bins=50, label='Nontarget')
    plt.hist(spoof_asv, histtype='step', density=True, bins=50, label='Spoof')
    plt.plot(asv_threshold, 0, 'o', markersize=10, mfc='none', mew=2, clip_on=False, label='EER threshold')
    plt.legend()
    plt.xlabel('ASV score')
    plt.ylabel('Density')
    plt.title('ASV score histogram')

    ax = plt.subplot(122)
    plt.hist(bona_cm, histtype='step', density=True, bins=50, label='Bona fide')
    plt.hist(spoof_cm, histtype='step', density=True, bins=50, label='Spoof')
    plt.legend()
    plt.xlabel('CM score')
    #plt.ylabel('Density')
    plt.title('CM score histogram')


    # Plot t-DCF as function of the CM threshold.
    plt.figure()
    plt.plot(CM_thresholds, tDCF_curve)
    plt.plot(CM_thresholds[min_tDCF_index], min_tDCF, 'o', markersize=10, mfc='none', mew=2)
    plt.xlabel('CM threshold index (operating point)')
    plt.ylabel('Norm t-DCF');
    plt.title('Normalized tandem t-DCF')
    plt.plot([np.min(CM_thresholds), np.max(CM_thresholds)], [1, 1], '--', color='black')
    plt.legend(('t-DCF', 'min t-DCF ({:.5f})'.format(min_tDCF), 'Arbitrarily bad CM (Norm t-DCF=1)'))
    plt.xlim([np.min(CM_thresholds), np.max(CM_thresholds)])
    plt.ylim([0, 1.5])

    plt.show()
