class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
       std::map<int,int> resd;
        vector<int> num;
         for(int i = 0 ; i < nums.size() ; i ++){
       
            int complement = target - nums[i];
            if(resd.find(complement) != resd.end()){
                num.push_back(resd[complement]);
                num.push_back(i);
            }

            resd[nums[i]] = i;
        }
        return num;
    }
};