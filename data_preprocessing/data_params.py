AU_INT = 'AU_INT'
AU_OCC = 'AU_OCC'
BASE_GROUND_PATH = 'C:/dataset/AUCoding'
BASE_DIR = 'C:/dataset/COLOR'
BAD_FILES_PATH = 'C:/dataset/badFiles_BP4D.txt'
IMAGE_TYPE = '.png'

class DataParams:
    def __init__(self):

        self.allSubjects_BP = ['F001',
			   'F002',
			   'F003',
			   'F004',
			   'F005',
			   'F006',
			   'F007',
			   'F008',
			   'F009',
			   'F010',
			   'F011',
			   'F012',
			   'F013',
			   'F014',
			   'F015',
			   'F016',
			   'F017',
			   'F018',
			   'F019',
			   'F020',
			   'F021',
			   'F022',
			   'F023',
			   'M001',
			   'M002',
			   'M003',
			   'M004',
			   'M005',
			   'M006',
			   'M007',
			   'M008',
			   'M009',
			   'M010',
			   'M011',
			   'M012',
			   'M013',
			   'M014',
			   'M015',
			   'M016',
			   'M017',
			   'M018']

        self.trainSubjects_BP = ['F001', 'F003', 'F005', 'F007', 'F009', 'F011', 'F013', 'F015', 'F019', 'F021', 'F023', 
					'M001', 'M003', 'M005', 'M007', 'M009', 'M011', 'M013', 'M015', 'M017']
					
        self.testSubjects_BP = ['F002', 'F004', 'F006', 'F008', 'F010', 'F012', 'F014', 'F016', 'F018', 'F020', 'F022',
					'M002', 'M004', 'M006', 'M008', 'M010', 'M012', 'M014', 'M016', 'M018']

        self.allIntAUs_BP = ['AU06','AU10','AU12','AU14','AU17']

        self.allTasks_BP = ['T1','T2','T3','T4','T5','T6','T7','T8']

        self.allOccAUs_BP = ["AU01",
				 "AU02",
				 "AU04",
				 "AU06",
				 "AU07",
				 "AU10",
				 "AU12",
				 "AU14",
				 "AU15",
				 "AU17",
				 "AU23",
				 "AU24"
				 ]

        self.lost_files = []

        self.groundPath = BASE_GROUND_PATH
        self.base_dir = BASE_DIR