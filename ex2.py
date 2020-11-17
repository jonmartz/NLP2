from language_model import Ngram_Language_Model, normalize_text
import math


class Spell_Checker:
    """The class implements a context sensitive spell checker. The corrections
        are done in the Noisy Channel framework, based on a language model and
        an error distribution model.
    """

    def __init__(self, lm=None):
        """Initializing a spell checker object with a language model as an
        instance  variable. The language model should suppport the evaluate()
        and the get_model() functions as defined in assignment #1.

        Args:
            lm: a language model object. Defaults to None
        """
        self.error_tables = None
        self.vocabulary = None
        self.char_counts = None
        self.two_char_counts = None
        self.lm = lm
        if lm is not None:
            self.process_model()

    def build_model(self, text, n=3, log_base=math.e):
        """Returns a language model object built on the specified text. The language
            model should support evaluate() and the get_model() functions as defined
            in assignment #1.

            Args:
                text (str): the text to construct the model from.
                n (int): the order of the n-gram model (defaults to 3).

            Returns:
                A language model object
        """
        self.get_char_counts(text.lower())
        lm = Ngram_Language_Model(n, log_base=log_base)
        lm.build_model(normalize_text(text))
        return lm

    def add_language_model(self, lm):
        """Adds the specified language model as an instance variable.
            (Replaces an older LM disctionary if set)

            Args:
                ls: a language model object
        """
        self.lm = lm
        self.process_model()

    def process_model(self):
        """
        Process the language model obtained.
        """
        self.vocabulary = set(self.lm.n_grams_by_len[0].keys())
        # for computing noisy channel prob:
        # self.total_words = sum(count for _, count in self.lm.n_grams_by_len[0].items())

    def get_char_counts(self, text):
        """
        Extract the counts of single and double characters (if alpha) in train text (should be lowercase).
        :param text: text with only lowercase
        """
        self.char_counts, self.two_char_counts = {}, {}
        for i in range(len(text)):
            one_char = text[i]
            if one_char.isalpha():
                self.char_counts[one_char] = self.char_counts.get(one_char, 0) + 1
                if i < len(text) - 1:
                    two_char = text[i: i + 2]
                    if two_char.isalpha():
                        self.two_char_counts[two_char] = self.two_char_counts.get(two_char, 0) + 1

    def learn_error_tables(self, errors_file):
        """Returns a nested dictionary {str:dict} where str is in:
            <'deletion', 'insertion', 'transposition', 'substitution'> and the
            inner dict {str: int} represents the confution matrix of the
            specific errors, where str is a string of two characters mattching the
            row and culumn "indixes" in the relevant confusion matrix and the int is the
            observed count of such an error (computed from the specified errors file).
            Examples of such string are 'xy', for deletion of a 'y'
            after an 'x', insertion of a 'y' after an 'x'  and substitution
            of 'x' (incorrect) by a 'y'; and example of a transposition is 'xy' indicates the characters that are transposed.


            Notes:
                1. Ultimately, one can use only 'deletion' and 'insertion' and have
                    'substitution' and 'transposition' derived. Again,  we use all
                    four types explicitly in order to keep things simple.
            Args:
                errors_file (str): full path to the errors file. File format, TSV:
                                    <error>    <correct>


            Returns:
                A dictionary of confusion "matrices" by error type (dict).
        """

    def add_error_tables(self, error_tables):
        """ Adds the speficied dictionary of error tables as an instance variable.
            (Replaces an older value disctionary if set)

            Args:
                error_tables (dict): a dictionary of error tables in the format
                returned by  learn_error_tables()
        """
        self.error_tables = error_tables

    def spell_check(self, text, alpha):
        """ Returns the most probable fix for the specified text. Use a simple
            noisy channel model if the number of tokens in the specified text is
            smaller than the length (n) of the language model.

            Args:
                text (str): the text to spell check.
                alpha (float): the probability of keeping a lexical word as is.

            Return:
                A modified string (or a copy of the original if no corrections are made.)
        """
        text = normalize_text(text)
        words = text.split(' ')
        best_text, best_log_prob = text, float('-inf')
        candidates_per_idx = self.get_candidates(text, alpha)
        for word_idx, candidates in enumerate(candidates_per_idx):
            for candidate_word, channel_prob in candidates:
                candidate_text = ' '.join(words[:word_idx] + [candidate_word] + words[word_idx + 1:])
                prior_log_prob = self.evaluate(candidate_text)
                channel_log_prob = math.log(channel_prob, self.lm.log_base)
                candidate_log_prob = channel_log_prob + prior_log_prob
                if candidate_log_prob > best_log_prob:
                    best_text, best_log_prob = candidate_text, candidate_log_prob
        return best_text

    def evaluate(self, text):
        """Returns the log-likelihod of the specified text given the language
            model in use. Smoothing is applied on texts containing OOV words

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        return self.lm.evaluate(text)

    def get_candidates(self, text, alpha):
        """
        Get all the candidate corrections for the text, which are the words editions up to an edit distance of 2.
        :param text: to generate candidates for
        :param alpha: probability of observed word being the correct word
        :return: list=[word_1=[[edit, channel_prob], ...], ..., word_n=[...]] where n=|words_in_text|
        """
        candidates = []
        for word in text.split(' '):
            if word.isalpha():
                edits = {word: alpha}  # initialize with original word
                edits_dist_1 = self.get_edits(word, edits, 1 - alpha)
                edits = {**edits, **edits_dist_1}
                for edit, prob in edits_dist_1.items():
                    edits_dist_2 = self.get_edits(edit, edits, prob)
                    edits = {**edits, **edits_dist_2}
                candidates.append([[edit, prob] for edit, prob in edits.items()
                                   if prob > 0 and edit in self.vocabulary])
        return candidates

    def get_edits(self, word, existing_edits, prior, letters='abcdefghijklmnopqrstuvwxyz'):
        """
        Return all edits of distance 1 from word, with their respective channel probability p(word|edit).
        :param word: to edit
        :param existing_edits: to avoid generating repeated edits (going back to a smaller edit distance and such)
        :param prior: to multiply the calculated channel probability by
        :param letters: alphabetic characters to consider
        :return: dict={edit_1: prob(word|edit_1), ..., edit_n: prob(word|edit_n)}
        """
        edits = {}
        if prior > 0:
            for i in range(len(word) + 1):
                if i < len(word):
                    # e stands for 'edit'
                    e = word[:i] + word[i + 1:]  # error was insertion
                    if e not in existing_edits and e not in edits:
                        table = self.error_tables['insertion']
                        if i == 0:
                            edits[e] = prior * table['#' + word[i]] / self.lm.corpus_len  # count of '#' character
                        else:
                            edits[e] = prior * table[e[i - 1] + word[i]] / self.char_counts.get(e[i - 1], 1)
                if i < len(word) - 1:
                    e = word[:i] + word[i + 1] + word[i] + word[i + 2:]  # error was transposition
                    if e not in existing_edits and e not in edits:
                        table = self.error_tables['transposition']
                        edits[e] = prior * table[e[i] + e[i + 1]] / self.two_char_counts.get(e[i] + e[i + 1], 1)
                for letter in letters:
                    e = word[:i] + letter + word[i:]  # error was deletion
                    if e not in existing_edits and e not in edits:
                        table = self.error_tables['deletion']
                        if i == 0:
                            edits[e] = prior * table['#' + e[i]] / self.char_counts.get(e[i], 1)
                        else:
                            edits[e] = prior * table[e[i - 1] + e[i]] / self.two_char_counts.get(e[i - 1] + e[i], 1)
                    if i < len(word):
                        e = word[:i] + letter + word[i + 1:]  # error was substitution
                        if e not in existing_edits and e not in edits:
                            table = self.error_tables['substitution']
                            edits[e] = prior * table[word[i] + e[i]] / self.char_counts.get(e[i], 1)
        return edits


def who_am_i():  # this is not a class method
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Jonathan Martinez', 'id': '201095569', 'email': 'martijon@post.bgu.ac.il'}
