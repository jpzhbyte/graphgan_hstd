function [W, D, L]=creatLap(X,k,sigma) 
      X=X';
      options = [];
      options.NeighborMode = 'KNN';%ѡ���޼ල�Ľ��ڱ����㷨
      options.k = k;
      options.WeightMode = 'HeatKernel';%ʵ��KNN����
      options.t = sigma;

      W = (constructW(X, options));
      D = (diag(sum(W, 2)));
      L = D - W;
end