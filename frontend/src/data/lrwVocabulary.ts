export interface LRWWordMeta {
  word: string
  cluster: 'Bilabial' | 'Labiodental' | 'Alveolar' | 'Velar' | 'Vocalic'
  phonemes: string
  frames: number
  trainClips: number
  valClips: number
  testClips: number
}

// Representative sample of LRW-500 corpus categorized by visual phonetic onset
export const LRW_VOCABULARY: LRWWordMeta[] = [
  // Bilabials (P, B, M) - Visual closure
  { word: 'BECAUSE', cluster: 'Bilabial', phonemes: '/bɪˈkɒz/', frames: 29, trainClips: 980, valClips: 50, testClips: 50 },
  { word: 'BEFORE', cluster: 'Bilabial', phonemes: '/bɪˈfɔː/', frames: 29, trainClips: 960, valClips: 50, testClips: 50 },
  { word: 'BELIEVE', cluster: 'Bilabial', phonemes: '/bɪˈliːv/', frames: 29, trainClips: 920, valClips: 50, testClips: 50 },
  { word: 'BETWEEN', cluster: 'Bilabial', phonemes: '/bɪˈtwiːn/', frames: 29, trainClips: 990, valClips: 50, testClips: 50 },
  { word: 'BUSINESS', cluster: 'Bilabial', phonemes: '/ˈbɪznɪs/', frames: 29, trainClips: 1000, valClips: 50, testClips: 50 },
  { word: 'PEOPLE', cluster: 'Bilabial', phonemes: '/ˈpiːpl/', frames: 29, trainClips: 1000, valClips: 50, testClips: 50 },
  { word: 'POLICE', cluster: 'Bilabial', phonemes: '/pəˈliːs/', frames: 29, trainClips: 950, valClips: 50, testClips: 50 },
  { word: 'PUBLIC', cluster: 'Bilabial', phonemes: '/ˈpʌblɪk/', frames: 29, trainClips: 930, valClips: 50, testClips: 50 },
  { word: 'PROBLEM', cluster: 'Bilabial', phonemes: '/ˈprɒbləm/', frames: 29, trainClips: 970, valClips: 50, testClips: 50 },
  { word: 'MARKET', cluster: 'Bilabial', phonemes: '/ˈmɑːkɪt/', frames: 29, trainClips: 940, valClips: 50, testClips: 50 },
  { word: 'MILLION', cluster: 'Bilabial', phonemes: '/ˈmɪljən/', frames: 29, trainClips: 990, valClips: 50, testClips: 50 },
  { word: 'MORNING', cluster: 'Bilabial', phonemes: '/ˈmɔːnɪŋ/', frames: 29, trainClips: 910, valClips: 50, testClips: 50 },

  // Labiodentals (F, V) - Lower lip to upper teeth
  { word: 'FAMILY', cluster: 'Labiodental', phonemes: '/ˈfæməli/', frames: 29, trainClips: 950, valClips: 50, testClips: 50 },
  { word: 'FEDERAL', cluster: 'Labiodental', phonemes: '/ˈfɛdərəl/', frames: 29, trainClips: 900, valClips: 50, testClips: 50 },
  { word: 'FINANCIAL', cluster: 'Labiodental', phonemes: '/faɪˈnænʃəl/', frames: 29, trainClips: 960, valClips: 50, testClips: 50 },
  { word: 'FOLLOW', cluster: 'Labiodental', phonemes: '/ˈfɒləʊ/', frames: 29, trainClips: 910, valClips: 50, testClips: 50 },
  { word: 'FOREIGN', cluster: 'Labiodental', phonemes: '/ˈfɒrɪn/', frames: 29, trainClips: 880, valClips: 50, testClips: 50 },
  { word: 'FUTURE', cluster: 'Labiodental', phonemes: '/ˈfjuːtʃə/', frames: 29, trainClips: 940, valClips: 50, testClips: 50 },
  { word: 'VICTIM', cluster: 'Labiodental', phonemes: '/ˈvɪktɪm/', frames: 29, trainClips: 860, valClips: 50, testClips: 50 },
  { word: 'VIOLENCE', cluster: 'Labiodental', phonemes: '/ˈvaɪələns/', frames: 29, trainClips: 890, valClips: 50, testClips: 50 },

  // Alveolars (T, D, N, L, S) - Tongue tip contact
  { word: 'TALKING', cluster: 'Alveolar', phonemes: '/ˈtɔːkɪŋ/', frames: 29, trainClips: 920, valClips: 50, testClips: 50 },
  { word: 'TEACHER', cluster: 'Alveolar', phonemes: '/ˈtiːtʃə/', frames: 29, trainClips: 880, valClips: 50, testClips: 50 },
  { word: 'TECHNOLOGY', cluster: 'Alveolar', phonemes: '/tɛkˈnɒlədʒi/', frames: 29, trainClips: 930, valClips: 50, testClips: 50 },
  { word: 'DECISION', cluster: 'Alveolar', phonemes: '/dɪˈsɪʒən/', frames: 29, trainClips: 970, valClips: 50, testClips: 50 },
  { word: 'DIFFERENT', cluster: 'Alveolar', phonemes: '/ˈdɪfrənt/', frames: 29, trainClips: 990, valClips: 50, testClips: 50 },
  { word: 'NATIONAL', cluster: 'Alveolar', phonemes: '/ˈnæʃənəl/', frames: 29, trainClips: 1000, valClips: 50, testClips: 50 },
  { word: 'NETWORK', cluster: 'Alveolar', phonemes: '/ˈnɛtwɜːk/', frames: 29, trainClips: 910, valClips: 50, testClips: 50 },
  { word: 'LEADER', cluster: 'Alveolar', phonemes: '/ˈliːdə/', frames: 29, trainClips: 960, valClips: 50, testClips: 50 },
  { word: 'LITTLE', cluster: 'Alveolar', phonemes: '/ˈlɪtl/', frames: 29, trainClips: 990, valClips: 50, testClips: 50 },
  { word: 'SECURITY', cluster: 'Alveolar', phonemes: '/sɪˈkjʊərɪti/', frames: 29, trainClips: 980, valClips: 50, testClips: 50 },
  { word: 'SYSTEM', cluster: 'Alveolar', phonemes: '/ˈsɪstəm/', frames: 29, trainClips: 1000, valClips: 50, testClips: 50 },

  // Velars (K, G) - Back of tongue to soft palate
  { word: 'CAMPAIGN', cluster: 'Velar', phonemes: '/kæmˈpeɪn/', frames: 29, trainClips: 940, valClips: 50, testClips: 50 },
  { word: 'COMMUNITY', cluster: 'Velar', phonemes: '/kəˈmjuːnɪti/', frames: 29, trainClips: 960, valClips: 50, testClips: 50 },
  { word: 'COMPANY', cluster: 'Velar', phonemes: '/ˈkʌmpəni/', frames: 29, trainClips: 1000, valClips: 50, testClips: 50 },
  { word: 'COUNTRY', cluster: 'Velar', phonemes: '/ˈkʌntri/', frames: 29, trainClips: 1000, valClips: 50, testClips: 50 },
  { word: 'GOVERNMENT', cluster: 'Velar', phonemes: '/ˈɡʌvənmənt/', frames: 29, trainClips: 1000, valClips: 50, testClips: 50 },
  { word: 'GROWTH', cluster: 'Velar', phonemes: '/ɡrəʊθ/', frames: 29, trainClips: 920, valClips: 50, testClips: 50 },

  // Vocalic / Rounded (W, R, O, U, A) - Distinct aperture & lip rounding
  { word: 'ABOUT', cluster: 'Vocalic', phonemes: '/əˈbaʊt/', frames: 29, trainClips: 1000, valClips: 50, testClips: 50 },
  { word: 'ACTION', cluster: 'Vocalic', phonemes: '/ˈækʃən/', frames: 29, trainClips: 950, valClips: 50, testClips: 50 },
  { word: 'AMERICAN', cluster: 'Vocalic', phonemes: '/əˈmɛrɪkən/', frames: 29, trainClips: 980, valClips: 50, testClips: 50 },
  { word: 'AROUND', cluster: 'Vocalic', phonemes: '/əˈraʊnd/', frames: 29, trainClips: 960, valClips: 50, testClips: 50 },
  { word: 'WATER', cluster: 'Vocalic', phonemes: '/ˈwɔːtə/', frames: 29, trainClips: 970, valClips: 50, testClips: 50 },
  { word: 'WOMAN', cluster: 'Vocalic', phonemes: '/ˈwʊmən/', frames: 29, trainClips: 950, valClips: 50, testClips: 50 },
  { word: 'WORLD', cluster: 'Vocalic', phonemes: '/wɜːld/', frames: 29, trainClips: 1000, valClips: 50, testClips: 50 },
]
