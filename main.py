from language_model import normalize_text
from ex2 import Spell_Checker
from spelling_confusion_matrices import error_tables
import pickle

corpora = open('corpora/big.txt').read() + ' '
# corpora = 'A cat sat on the mat. A fat cat sat on the mat. A rat sat on the mat. The rat sat on the cat. A bat spat on the rat that sat on the cat on the mat. '

alpha = 0.9
# alpha = 0.5
texts = [
    # 'acress the sky',
    # 'she is an acress',
    'his volley acress the field was just glorious',
    # 'another acress accuses harvey weinstein of rape',
    'an acress accuses harvey weinstein of rape',
    # 'an acress',
    # 'the acress',
    # 'another acress',
    # 'another actress',
    # 'another across',
]

# print('building language model...')
# spell_checker = Spell_Checker()
# lm = spell_checker.build_model(corpora)
# spell_checker.add_language_model(lm)
# spell_checker.add_error_tables(error_tables)
# pickle.dump(spell_checker, open('model.sav', 'wb'))

spell_checker = pickle.load(open('model.sav', 'rb'))

for text in texts:
    print()
    print('original  = %s (prior = %s)' % (text, spell_checker.lm.evaluate(text)))
    corrected = spell_checker.spell_check(text, alpha)
    print('corrected = %s (prior = %s)' % (corrected, spell_checker.lm.evaluate(corrected)))
