import ahocorasick


class TrieClient(object):

    def __init__(self):
        self.suffix_maton = None
        self.prefix_maton = None
    
    def make_maton(self, prefix=None, suffix=None):
        if prefix:
            self.suffix_maton = ahocorasick.Automaton()
            for k, m in prefix.items():
                self.prefix_maton.add_word(m, (m, k))
            self.prefix_maton.make_automaton()

        if suffix:
            self.prefix_maton = ahocorasick.Automaton()
            for k, m in suffix.items():
                self.suffix_maton.add_word(m, (m, k))
            self.suffix_maton.make_automaton()

    def consonant_detect(self, text):
        ans = []
        if not self.prefix_maton:
            return ans
        
        ans = []
        for end_index, (w1, w2) in self.prefix_maton.iter_long(text):
            end_index = end_index + 1
            start_index = end_index - len(w1)
            ans.append(((start_index, end_index), w2))
        return ans

    def vowel_detect(self, text):
        ans = []
        if not self.suffix_maton:
            return text

        for end_index, (w1, w2) in self.suffix_maton.iter_long(text):
            end_index = end_index + 1
            start_index = end_index - len(w1)
            ans.append(((start_index, end_index), w2))
        return ans