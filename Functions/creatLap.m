function [W, D, L]=creatLap(X,k,sigma) 
      X=X';
      options = [];
      options.NeighborMode = 'KNN';%选择无监督的近邻保持算法
      options.k = k;
      options.WeightMode = 'HeatKernel';%实现KNN聚类
      options.t = sigma;

      W = (constructW(X, options));
      D = (diag(sum(W, 2)));
      L = D - W;
end