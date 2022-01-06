"""
Bayes and Naive Bayes classifiers with Maximum Likelihood Estimation using a 
guassian PDF.
"""

import numpy as np
import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.model_selection import train_test_split


class Bayes:
    """
    Bayes classifier.
    """

    def __init__(self, num_targets: int):
        """Bayes classifier constructor.

        Parameters
        ----------
        num_targets : int
            Number of target values in the dataset.
        """
        self.num_targets = num_targets
        self.pdf = list()

    def _build_pdf(self, df: pd.DataFrame, diag_cov: bool = False,) -> list:
        """Build a gaussian PDF for each class in the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        diag_cov : bool, optional
            Whether to use diagonal covariance matrix or not, by default False.

        Returns
        -------
        list
            A list of gaussian PDF functions for each class. List index corresponds to the class number.
        """
        pdf = list()

        # make a pdf for each class
        for C in range(self.num_targets):
            # caculate the covariance between features
            cov = df[df["target"] == C].drop(columns=["target"]).cov().to_numpy()

            # calculate the mean of each feature
            mean = (
                df[df["target"] == C]
                .drop(columns=["target"])
                .mean()
                .to_numpy()
                .reshape((-1, 1))
            )

            # make the cov diagonal if needed
            if diag_cov:
                cov = np.diag(np.diag(cov))

            # make a pdf lambda function with the mean and cov
            pdf.append(
                lambda X, cov=cov, mean=mean: (
                    1 / ((2 * np.pi) ** ((df.shape[1] - 1) / 2))
                )
                * (1 / (np.linalg.det(cov) ** 0.5))
                * np.exp(
                    -0.5
                    * (X.reshape((-1, 1)) - mean).T
                    @ np.linalg.inv(cov)
                    @ (X.reshape((-1, 1)) - mean)
                ).item()
            )

        return pdf

    @staticmethod
    def _random_split(
        df: pd.DataFrame, train_frac: float, seed: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Randomly split the given DataFrame using the train_frac parameter.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        train_frac : float
            Fraction of training data; fraction of test data will be (1 - train_frac).
        seed : int, optional
            The seed to use for random_state parameter in sklearn's train_test_split, by default None

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (train_dataframe, test_dataframe)
        """
        # split the features and the targets
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1].to_frame()

        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, train_size=train_frac, random_state=seed,
        )

        # merge the features and the targets again
        df_train = pd.concat((X_train, y_train), axis=1).reset_index(drop=True)
        df_test = pd.concat((X_test, y_test), axis=1).reset_index(drop=True)

        return df_train, df_test

    def train(
        self,
        df: pd.DataFrame,
        kfold: bool = False,
        train_frac: float = None,
        iterations: int = 2,
        diag_cov: bool = False,
    ):
        """Train a bayes classifier with the given data. The accuracy and the 
        confusion matrix will be shown after training. The trained gaussian PDF 
        functions will be saved in self.pdf variable.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        kfold : bool, optional
            Whether to perform k-fold cross validation or not, by default False.
        train_frac : float, optional
            Fraction of the data to be used as training data, by default None.
        iterations : int, optional
            If kfold is True, this will be the parameter k; otherwise, the data 
            will be randomly splitted into train and test and trained for number 
            of iterations, by default 2.
        diag_cov : bool, optional
            Whether to use a diagonal covariance matrix or not, by default False
        """
        # initialize the evaluation parameters
        train_conf_mat = list()
        train_acc = list()
        test_conf_mat = list()
        test_acc = list()

        # size of each column when printing the output table
        cols = [6, 7, 4]

        # set the first column title
        if kfold:
            col_0 = "fold"
        else:
            col_0 = "iter"

        # print the table's header
        print(
            f"{col_0:<{cols[0]}}",
            f"{'train':<{cols[1]}}",
            f"{'test':<{cols[2]}}",
            sep="",
        )
        print("=" * sum(cols))

        # get 'kfold' or 'random splitting' for-loop-generator
        if kfold:
            # build a stratified kfold object
            skf = sklearn.model_selection.StratifiedKFold(
                n_splits=iterations, shuffle=False,
            )

            # split the features and the targets
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1].to_frame()

            for_gen = skf.split(X, y)
        else:
            for_gen = range(iterations)

        # perform the 'kfold' or 'random splitting' cross validation
        for i, item in enumerate(for_gen):
            # do the splitting
            if kfold:
                # split the data into train and test
                X_train, X_test = X.iloc[item[0]], X.iloc[item[1]]
                y_train, y_test = y.iloc[item[0]], y.iloc[item[1]]

                # re-merge the features and the targets
                train = pd.concat((X_train, y_train), axis=1,).reset_index(drop=True)
                test = pd.concat((X_test, y_test), axis=1,).reset_index(drop=True)
            else:
                train, test = self._random_split(df, train_frac, seed=i)

            # create the pdf
            self.pdf = self._build_pdf(train, diag_cov=diag_cov)

            # predict the training dataset targets
            y_train_true = train.iloc[:, -1].to_numpy()
            y_train_pred = self.predict(train.iloc[:, :-1].to_numpy())

            # predict the testing dataset targets
            y_test_true = test.iloc[:, -1].to_numpy()
            y_test_pred = self.predict(test.iloc[:, :-1].to_numpy())

            # evaluate the training and the testing datasets
            train_conf_mat_, train_acc_ = self.evaluate(y_train_true, y_train_pred)
            test_conf_mat_, test_acc_ = self.evaluate(y_test_true, y_test_pred)

            # print the evaluations
            print(
                f"{(i+1):<{cols[0]}}",
                f"{train_acc_:<{cols[1]}.2f}",
                f"{test_acc_:<{cols[2]}.2f}",
                sep="",
            )

            # append to the metrics variables
            train_conf_mat.append(train_conf_mat_)
            test_conf_mat.append(test_conf_mat_)
            train_acc.append(train_acc_)
            test_acc.append(test_acc_)

        # calculate the average metrics
        train_conf_mat = np.average(train_conf_mat, axis=0)
        test_conf_mat = np.average(test_conf_mat, axis=0)
        train_acc = sum(train_acc) / len(train_acc)
        test_acc = sum(test_acc) / len(test_acc)

        # print the average metrics
        print(
            f"{'avg':<{cols[0]}}",
            f"{train_acc:<{cols[1]}.2f}",
            f"{test_acc:<{cols[2]}.2f}",
            sep="",
            end="\n\n",
        )

        # plot the confusion matrix
        self.plot_conf_mat(
            train_conf_mat,
            test_conf_mat,
            "Normalized Train Confusion Matrix",
            "Normalized Test Confusion Matrix",
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the targets based on the trained self.pdf parameter.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            An array of predicted targets for each sample.
        """
        y_pred = list()

        # caculate probabilities for each pdf
        for pdf_ in self.pdf:
            y_pred.append(list(map(pdf_, X)))

        # apply argmax to the probabilities to get the predicted target
        y_pred = np.asarray(y_pred).argmax(axis=0)

        return y_pred

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
        """Calculate the confusion matrix and the accuracy with the given y_true 
        and y_ped.

        Parameters
        ----------
        y_true : np.ndarray
            An array of the true targets.
        y_pred : np.ndarray
            An array of the predicted targets.

        Returns
        -------
        Tuple[np.ndarray, float]
            (Confusion matrix, Accuracy)
        """
        # create the confusion matrix
        conf_mat = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize="true")

        # caculate the accuracy
        accuracy = sum(conf_mat.diagonal()) / len(conf_mat.diagonal())

        return conf_mat, accuracy

    @staticmethod
    def plot_conf_mat(
        train_conf_mat: np.ndarray,
        test_conf_mat: np.ndarray,
        train_title: str = "",
        test_title: str = "",
    ):
        """Plot the two confusion matrices next to each other.

        Parameters
        ----------
        train_conf_mat : np.ndarray
            The trainig dataset confusion matrix.
        test_conf_mat : np.ndarray
            The testing dataset confusion matrix.
        train_title : str, optional
            The title for the training dataset confusion matrix, by default "".
        test_title : str, optional
            The title for the testing dataset confusion matrix, by default "".
        """
        # initialize the plot
        fig, ax = plt.subplots(1, 2)
        fig.set_figheight(2 * 3)
        fig.set_figwidth(3 * 4)

        # create the plots
        sklearn.metrics.ConfusionMatrixDisplay(train_conf_mat).plot(
            ax=ax[0], colorbar=False,
        )
        sklearn.metrics.ConfusionMatrixDisplay(test_conf_mat).plot(
            ax=ax[1], colorbar=False,
        )

        # set plot titles
        ax[0].set_title(train_title)
        ax[1].set_title(test_title)

        # show the plot
        fig.show()


class NaiveBayes(Bayes):
    """
    Naive Bayes classifier.
    """

    def __init__(self, num_targets: int):
        """Naive Bayes classifier constructor.

        Parameters
        ----------
        num_targets : int
            Number of target values in the dataset.
        """
        super().__init__(num_targets)

    def _build_pdf(self, df: pd.DataFrame, *args, **kwargs,) -> list:
        """Build a gaussian PDF for each class and feature in the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.

        Returns
        -------
        list
            A list of gaussian PDF functions for each class and feature. List 
            index corresponds to the class number.
        """
        pdf = list()

        # make a pdf for each class and each feature
        for C in range(self.num_targets):
            df_c = df[df["target"] == C].drop(columns=["target"])
            pdf.append([])
            for F in range(df.shape[1] - 1):
                # caculate the variance of the feature
                var = df_c.iloc[:, F].var()

                # calculate the mean of the feature
                mean = df_c.iloc[:, F].mean()

                # make a pdf lambda function with the mean and var
                pdf[C].append(
                    lambda x, var=var, mean=mean: (1 / (2 * np.pi * var) ** 0.5)
                    * np.exp(-(0.5 / var) * (x - mean) ** 2)
                )

        return pdf

    def train(
        self,
        df: pd.DataFrame,
        kfold: bool = False,
        train_frac: float = None,
        iterations: int = 2,
        *args,
        **kwargs,
    ):
        """Train a naive bayes classifier with the given data. The accuracy and 
        the confusion matrix will be shown after training. The trained gaussian 
        PDF functions will be saved in self.pdf variable.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        kfold : bool, optional
            Whether to perform k-fold cross validation or not, by default False.
        train_frac : float, optional
            Fraction of the data to be used as training data, by default None.
        iterations : int, optional
            If kfold is True, this will be the parameter k; otherwise, the data 
            will be randomly splitted into train and test and trained for number 
            of iterations, by default 2.
        """
        super().train(df, kfold, train_frac, iterations, diag_cov=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the targets based on the trained self.pdf parameter.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            An array of predicted targets for each sample.
        """
        y_pred = list()

        # iterate on all samples
        for x in X:
            # initialize a list of probabilities for each target value
            y_pred_ = [1] * self.num_targets
            # iterate on the pdf functions for each target
            for (idx, pdf_) in enumerate(self.pdf):
                # iterate on each feature and their pdf function
                for (x_, pdf__) in zip(x, pdf_):
                    # calculate the probability for the current target given the
                    # current feature
                    y_pred_[idx] *= pdf__(x_)
            # append the calculated probabilities to the y_pred list
            y_pred.append(y_pred_)

        # apply argmax to the probabilities to get the predicted target
        y_pred = np.asarray(y_pred).argmax(axis=1)

        return y_pred
