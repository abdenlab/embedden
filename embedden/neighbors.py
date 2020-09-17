"""
scikit-learn NearestNeighbors
=============================
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neighbors/_unsupervised.py

spotify annoy
=============
https://github.com/spotify/annoy

pynndescent
===========
https://github.com/lmcinnes/pynndescent

"""
from sklearn.base import BaseEstimator, TransformerMixin


class SKLearnNeighbors(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        n_neighbors=15,
        metric='minkowski',
        p=2,
        metric_params=None,
        radius=1.0,
        algorithm='auto',
        leaf_size=30,
        n_jobs=None
    ):
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            raise ValueError("scikit-learn is required")

        self.model = NearestNeighbors(
            n_neighbors=n_neighbors,
            radius=radius,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )

    def fit(self, X):
        # fit only constructs the tree, doesn't do the search
        self.model.fit(X)
        # this does the full search
        dists, inds = self.model.kneighbors(X)
        self.knn_indices_ = inds
        self.knn_dists_ = dists
        return self

    def transform(self, X):
        return self.knn_indices_, self.knn_dists_


class PyNNDescentNeighbors(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        n_neighbors=15,
        metric="euclidean",
        metric_kwds=None,
        n_trees=None,
        leaf_size=None,
        pruning_degree_multiplier=2.0,
        diversify_epsilon=1.0,
        n_search_trees=1,
        tree_init=True,
        random_state=None,
        algorithm="standard",
        low_memory=False,
        max_candidates=None,
        n_iters=None,
        delta=0.001,
        n_jobs=None,
        compressed=False,
        seed_per_row=False,
        verbose=False
    ):
        try:
            import pynndescent
        except ImportError:
            raise ValueError("The pynndescent package is required")

        self._kwargs = dict(
            metric=metric,
            metric_kwds=metric_kwds,
            n_neighbors=n_neighbors,
            n_trees=n_trees,
            leaf_size=leaf_size,
            pruning_degree_multiplier=pruning_degree_multiplier,
            diversify_epsilon=diversify_epsilon,
            n_search_trees=n_search_trees,
            tree_init=tree_init,
            random_state=random_state,
            algorithm=algorithm,
            low_memory=low_memory,
            max_candidates=max_candidates,
            n_iters=n_iters,
            delta=delta,
            n_jobs=n_jobs,
            compressed=compressed,
            seed_per_row=seed_per_row,
            verbose=verbose
        )

    def fit(self, X):
        from pynndescent import NNDescent
        # This constructs the index and does the search
        self.nnd = NNDescent(X, **self._kwargs)
        self.knn_indices_, self.knn_dists_ = self.nnd.neighbor_graph
        return self

    def transform(self, X):
        return self.knn_indices_, self.knn_dists_


class AnnoyNeighbors(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        n_neighbors=15,
        metric="euclidean",
        n_trees=10,
        n_jobs=1,
        search_k=-1,
        seed=None,
        on_disk_build=False,
    ):
        try:
            import annoy
        except ImportError:
            raise ValueError("The annoy package is required")

        self._n_neighbors = n_neighbors
        self._metric = metric
        self._n_trees = n_trees
        self._n_jobs = n_jobs
        self._search_k = search_k
        self._seed = seed
        self._on_disk_build = on_disk_build

    def fit(self, X):

        from annoy import AnnoyIndex

        self.index = AnnoyIndex(X.shape[1], self._metric)

        if self._seed is not None:
            self.index.set_seed(self._seed)

        a = self.index
        n = self._n_neighbors
        search_k = self._search_k

        # If building on disk, set the index file before adding items
        if self._on_disk_build:
            a.on_disk_build(self._on_disk_build)

        # Populate the data structure
        for i in range(X.shape[0]):
            a.add_item(i, X[i, :])

        # If building in RAM, build the index after all items are added
        if not self._on_disk_build:
            a.build(self._n_trees, self._n_jobs)

        # Do the search
        knn_indices = []
        knn_dists = []
        for i in range(X.shape[0]):
            inds, dists = a.get_nns_by_item(
                i, n, search_k=search_k, include_distances=True
            )
            knn_indices.append(inds)
            knn_dists.append(dists)
        self.knn_indices_ = np.array(knn_indices)
        self.knn_dists_ = np.array(knn_dists)

        return self

    def transform(self, X):
        return self.knn_indices_, self.knn_dists_
