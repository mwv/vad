#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: _vad.py
# date: Thu October 23 00:38 2014
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""Voice Activity Detector
This is a straight-forward re-implementation of Bowon Lee's Voice Activity
Detector. The relevant paper to cite is: Bowon Lee and Mark Hasegawa-Johnson,
"Minimum Mean-squared Error A Posteriori Estimation of High Variance Vehicular
Noise". Please cite that paper when using this code for research purposes.

Usage:
>>> from vad import VAD
>>> detector = VAD()
>>> speech = detector.detect_speech(sig, fs)

"""

from __future__ import division

import numpy as np

# some underflow falls through after calls to i0 and i1, but these are caught
# by putmask before updating G_MMSE. errors are silenced to not bother the user
np.seterr(all='ignore')

from numpy import sqrt, minimum, maximum, exp, pi, hamming, floor, zeros, \
    conj, isnan, log
from numpy.lib.stride_tricks import as_strided
from scipy.special import i0, i1


class VAD(object):
    def __init__(self, fs, markov_params=(0.5, 0.1), alpha=0.99, NFFT=2048,
                 n_iters=10, win_size_sec=0.05, win_hop_sec=0.025,
                 max_est_iter=-1, epsilon=1e-6):
        """

        Arguments:
        :param markov_params: hangover scheme params
        :param alpha: SNR estimate coefficient
        :param NFFT: size of FFT
        :param n_iters: number of iterations in noise estimation
        :param win_size_sec: window size in seconds
        :param win_hop_sec: hop size in seconds
        :param epsilon: convergence epsilon
        """
        self.fs = fs
        self.a01, self.a10 = markov_params
        self.a00 = 1 - self.a01
        self.a11 = 1 - self.a10
        self.alpha = alpha
        self.NFFT = NFFT
        self.win_size_sec = win_size_sec
        self.win_hop_sec = win_hop_sec
        self.epsilon = epsilon
        self.n_iters = n_iters
        self.max_est_iter = max_est_iter

        self.wlen = int(fs * self.win_size_sec)
        self.fshift = int(fs * self.win_hop_sec)
        self.win = hamming(self.wlen)

    def detect_speech(self, sig, fs, threshold, n_noise_frames=20):
        return self.activations(sig, fs, n_noise_frames) > threshold

    def stft(self, sig):
        s = np.pad(sig, (self.wlen//2, 0), 'constant')
        cols = np.ceil((s.shape[0] - self.wlen) / self.fshift + 1)
        s = np.pad(s, (0, self.wlen), 'constant')
        frames = as_strided(s, shape=(cols, self.wlen),
                            strides=(s.strides[0]*self.fshift,
                                     s.strides[0])).copy()
        return np.fft.rfft(frames*self.win, self.NFFT)

    def activations(self, sig, fs, n_noise_frames=20):
        frames = self.stft(sig)
        n_frames = frames.shape[0]

        noise_var_tmp = zeros(self.NFFT//2+1)
        for n in xrange(n_noise_frames):
            frame = frames[n]
            noise_var_tmp = noise_var_tmp + (conj(frame) * frame).real

        noise_var_orig = noise_var_tmp / n_noise_frames
        noise_var_old = noise_var_orig

        G_old = 1
        A_MMSE = zeros((self.NFFT//2+1, n_frames))
        G_MMSE = zeros((self.NFFT//2+1, n_frames))

        cum_Lambda = zeros(n_frames)
        for n in xrange(n_frames):
            frame = frames[n]
            frame_var = (conj(frame) * frame).real

            noise_var = noise_var_orig

            if self.max_est_iter == -1 or n < self.max_est_iter:
                noise_var_prev = noise_var_orig
                for iter_idx in xrange(self.n_iters):
                    gamma = frame_var / noise_var
                    Y_mag = np.abs(frame)

                    if n:
                        xi = (self.alpha *
                              ((A_MMSE[:, n-1]**2 / noise_var_old) +
                               (1 - self.alpha) * maximum(gamma - 1, 0)))
                    else:
                        xi = (self.alpha +
                              (1 - self.alpha) * maximum(gamma - 1, 0))
                    v = xi * gamma / (1 + xi)
                    bessel_1 = i1(v/2)
                    bessel_0 = i0(v/2)
                    g_upd = (sqrt(pi) / 2) * (sqrt(v) / gamma) * np.exp(v/-2) * \
                        ((1 + v) * bessel_0 + v * bessel_1)
                    np.putmask(g_upd, np.logical_not(np.isfinite(g_upd)), 1.)
                    G_MMSE[:, n] = g_upd
                    A_MMSE[:, n] = G_MMSE[:, n] * Y_mag

                    gamma_term = gamma * xi / (1 + xi)
                    gamma_term = minimum(gamma_term, 1e-2)
                    Lambda_mean = (1 / (1 + xi) + exp(gamma_term)).mean()

                    weight = Lambda_mean / (1 + Lambda_mean)
                    if isnan(weight):
                        weight = 1

                    noise_var = weight * noise_var_orig + (1 - weight) * frame_var

                    diff = np.abs(np.sum(noise_var - noise_var_prev))
                    if diff < self.epsilon:
                        break
                    noise_var_prev = noise_var

            gamma = frame_var / noise_var
            Y_mag = np.abs(frame)

            if n:
                xi = self.alpha * ((A_MMSE[:, n-1]**2 / noise_var_old) +
                                   (1 - self.alpha) * maximum(gamma - 1, 0))
            else:
                xi = self.alpha + (1 - self.alpha) * maximum(gamma - 1, 0)

            v = (xi * gamma) / (1 + xi)
            bessel_0 = i0(v/2)
            bessel_1 = i1(v/2)
            g_upd = (sqrt(pi) / 2) * (sqrt(v) / gamma) * \
                exp(v/-2) * ((1 + v) * bessel_0 + v * bessel_1)
            np.putmask(g_upd, np.logical_not(np.isfinite(g_upd)), 1.)
            G_MMSE[:, n] = g_upd
            A_MMSE[:, n] = G_MMSE[:, n] * Y_mag

            Lambda_mean = (log(1/(1+xi)) + gamma * xi / (1 + xi)).mean()

            G = ((self.a01 + self.a11 * G_old) /
                 (self.a00 + self.a10 * G_old) * Lambda_mean)

            cum_Lambda[n] = G

            G_old = G
            noise_var_old = noise_var
        return cum_Lambda
