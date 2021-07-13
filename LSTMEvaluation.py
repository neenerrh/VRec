import heapq
import numpy as np 

class LSTMEvaluation(object):
	'''
	recurrent neural network evaluation
	'''
	def __init__(self, embedding_dict, all_movie, train_dict, test_dict):
		super(LSTMEvaluation, self).__init__()
		self.embedding_dict = embedding_dict
		self.all_movie = all_movie
		self.train_dict = train_dict
		self.test_dict = test_dict
		self.mrr = 0.0
		self.map=0.0


	def calculate_ranking_score(self):
		'''
		calculate ranking score of unrated movies for each user
		'''
		score_dict = {}
		top_score_dict = {}

		for user in self.test_dict:
			if user in self.embedding_dict and user in self.train_dict:
				for movie in self.all_movie:
					if movie not in self.train_dict[user] and movie in self.embedding_dict:
						embedding_user = self.embedding_dict[user]
						embedding_movie = self.embedding_dict[movie]
						score = np.dot(embedding_user, embedding_movie)
						if user not in score_dict:
							score_dict.update({user:{movie:score}})
						else:
							score_dict[user].update({movie:score})

				#rank score in a descending order
				if user in score_dict and len(score_dict[user]) > 1:
					item_score_list = score_dict[user]
					k = min(len(item_score_list), 15) #to speed up the ranking process, we only find the top 15 movies
					top_item_list = heapq.nlargest(k, item_score_list, key=item_score_list.get)
					top_score_dict.update({user:top_item_list})

		return top_score_dict

	def calculate_results(self, top_score_dict, k):
		'''
		calculate the final results: pre@k and mrr
		'''
		precision = 0.0
		recall=0.0
		isMrr = False
		isMap = False
		if k == 10:
			isMrr = True
			isMap = True
		
		user_size = 0
		for user in self.test_dict:
			if user in top_score_dict:
				user_size = user_size + 1
				candidate_item = top_score_dict[user]
				candidate_size = len(candidate_item)
				hit = 0
				sum_prec=0.0

				min_len = min(candidate_size, k)
				for i in range(min_len):
					if candidate_item[i] in self.test_dict[user]:
						hit = hit + 1
						if isMrr: self.mrr += float(1/(i+1))
						if isMap: sum_precs += float(hit /(i+1.0))
				if hit > 0:
					self.map += sum_precs/ len(self.test_dict[user])
				else:
					self.map += 0.0
				hit_ratio = float(hit / min_len)
				hit_ratio2 = float(hit/len(self.test_dict[user]))
				precision += hit_ratio
				recall +=  hit_ratio2
		precision = precision / user_size
		recall = recall/user_size
		print ('precision@' + str(k) + ' is: ' + str(precision))
		print ('recall@' + str(k) + ' is: ' + str(recall))
		if isMrr:
			self.mrr = self.mrr / user_size
			print ('mrr@' + str(k) +' is: ' + str(self.mrr))
		if isMap:
			self.map=self.map/user_size
			print('map@'+str(k) +' is: "+str(self.map)
		return precision, self.mrr, self.map ,recall
	
	        
