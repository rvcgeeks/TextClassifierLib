#include <string>
#include <vector>

using namespace std;

class MyGlobalVars{
public:
   int POS;
   int NEG;
   int NEU;
   int UNK;
   vector<string> sent1;
   vector<string> sent2;
   vector<string> features;
   vector<bool> labels;

  MyGlobalVars(){
     POS = 1;
     NEG = 0;
     NEU = -2;
     UNK = 0;
     sent1 = {"This", "is", "my", "new", "sentence"};
     sent2 = {"this", "cat", "!"};
     features = {
      "Wow... Loved this place.",
      "Crust is not good.",
      "Not tasty and the texture was just nasty.",
      "Stopped by during the late May bank holiday off Rick Steve recommendation and loved it.",
      "The selection on the menu was great and so were the prices.",
      "Now I am getting angry and I want my damn pho.",
      "Honeslty it didn't taste THAT fresh.)",
      "The potatoes were like rubber and you could tell they had been made up ahead of time being kept under a warmer.",
      "The fries were great too.",
      "A great touch.",
      "Service was very prompt.",
      "Would not go back.",
      "The cashier had no care what so ever on what I had to say it still ended up being wayyy overpriced.",
      "I tried the Cape Cod ravoli, chicken,with cranberry...mmmm!",
      "I was disgusted because I was pretty sure that was human hair.",
      "I was shocked because no signs indicate cash only.",
      "Highly recommended.",
      "Waitress was a little slow in service.",
      "This place is not worth your time, let alone Vegas.",
      "did not like at all.",
      "The Burrittos Blah!",
      "The food, amazing.",
      "Service is also cute.",
      "I could care less... The interior is just beautiful.",
      "So they performed.",
      "That's right....the red velvet cake.....ohhh this stuff is so good.",
      "- They never brought a salad we asked for.",
      "This hole in the wall has great Mexican street tacos, and friendly staff.",
      "Took an hour to get our food only 4 tables in restaurant my food was Luke warm, Our sever was running around like he was totally overwhelmed.",
      "The worst was the salmon sashimi.",
      "Also there are combos like a burger, fries, and beer for 23 which is a decent deal.",
      "This was like the final blow!",
      "I found this place by accident and I could not be happier.",
      "seems like a good quick place to grab a bite of some familiar pub food, but do yourself a favor and look elsewhere.",
      "Overall, I like this place a lot.",
      "The only redeeming quality of the restaurant was that it was very inexpensive.",
      "Ample portions and good prices.",
      "Poor service, the waiter made me feel like I was stupid every time he came to the table.",
      "My first visit to Hiro was a delight!",
      "Service sucks.",
      "The shrimp tender and moist.",
      "There is not a deal good enough that would drag me into that establishment again.",
      "Hard to judge whether these sides were good because we were grossed out by the melted styrofoam and didn't want to eat it for fear of getting sick.",
      "On a positive note, our server was very attentive and provided great service.",
      "Frozen pucks of disgust, with some of the worst people behind the register.",
      "The only thing I did like was the prime rib and dessert section.",
      "It's too bad the food is so damn generic.",
      "The burger is good beef, cooked just right.",
      "If you want a sandwich just go to any Firehouse!!!!!",
      "My side Greek salad with the Greek dressing was so tasty, and the pita and hummus was very refreshing.",
      "We ordered the duck rare and it was pink and tender on the inside with a nice char on the outside.",
      "He came running after us when he realized my husband had left his sunglasses on the table.",
      "Their chow mein is so good!",
      "They have horrible attitudes towards customers, and talk down to each one when customers don't enjoy their food.",
      "The portion was huge!",
      "Loved it...friendly servers, great food, wonderful and imaginative menu.",
      "The Heart Attack Grill in downtown Vegas is an absolutely flat-lined excuse for a restaurant.",
      "Not much seafood and like 5 strings of pasta at the bottom.",
      "The salad had just the right amount of sauce to not over power the scallop, which was perfectly cooked.",
      "The ripped banana was not only ripped, but petrified and tasteless.",
      "At least think to refill my water before I struggle to wave you over for 10 minutes.",
      "This place receives stars for their APPETIZERS!!!",
      "The cocktails are all handmade and delicious.",
      "We'd definitely go back here again.",
      "We are so glad we found this place.",
      "Great food and service, huge portions and they give a military discount.",
      "Always a great time at Dos Gringos!",
      "Update.....went back for a second time and it was still just as amazing",
      "We got the food and apparently they have never heard of salt and the batter on the fish was chewy.",
      "A great way to finish a great.",
      "The deal included 5 tastings and 2 drinks, and Jeff went above and beyond what we expected.",
      "- Really, really good rice, all the time.",
      "The service was meh.",
      "It took over 30 min to get their milkshake, which was nothing more than chocolate milk.",
      "I guess I should have known that this place would suck, because it is inside of the Excalibur, but I didn't use my common sense.",
      "The scallop dish is quite appalling for value as well.",
      "2 times - Very Bad Customer Service !",
      "The sweet potato fries were very good and seasoned well.",
      "Today is the second time I've been to their lunch buffet and it was pretty good.",
      "There is so much good food in Vegas that I feel cheated for wasting an eating opportunity by going to Rice and Company.",
      "Coming here is like experiencing an underwhelming relationship where both parties can't wait for the other person to ask to break up.",
      "walked in and the place smelled like an old grease trap and only 2 others there eating.",
      "The turkey and roast beef were bland.",
      "This place has it!",
      "The pan cakes everyone are raving about taste like a sugary disaster tailored to the palate of a six year old.",
      "I love the Pho and the spring rolls oh so yummy you have to try.",
      "The poor batter to meat ratio made the chicken tenders very unsatisfying.",
      "All I have to say is the food was amazing!!!",
      "Omelets are to die for!",
      "Everything was fresh and delicious!",
      "In summary, this was a largely disappointing dining experience.",
      "It's like a really sexy party in your mouth, where you're outrageously flirting with the hottest person at the party.",
      "Never been to Hard Rock Casino before, WILL NEVER EVER STEP FORWARD IN IT AGAIN!",
      "Best breakfast buffet!!!",
      "say bye bye to your tip lady!",
      "We'll never go again.",
      "Will be back again!",
      "Food arrived quickly!",
      "It was not good.",
      "On the up side, their cafe serves really good food.",
      "Our server was fantastic and when he found out the wife loves roasted garlic and bone marrow, he added extra to our meal and another marrow to go!",
      "The only good thing was our waiter, he was very helpful and kept the bloddy mary's coming.",
      "Best Buffet in town, for the price you cannot beat it.",
      "I LOVED their mussels cooked in this wine reduction, the duck was tender, and their potato dishes were delicious.",
      "This is one of the better buffets that I have been to.",
      "So we went to Tigerlilly and had a fantastic afternoon!",
      "The food was delicious, our bartender was attentive and personable AND we got a great deal!",
      "The ambience is wonderful and there is music playing.",
      "Will go back next trip out.",
      "Sooooo good!!",
      "REAL sushi lovers, let's be honest - Yama is not that good.",
      "At least 40min passed in between us ordering and the food arriving, and it wasn't that busy.",
      "This is a really fantastic Thai restaurant which is definitely worth a visit.",
      "Nice, spicy and tender.",
      "Good prices.",
      "Check it out.",
      "It was pretty gross!",
      "I've had better atmosphere.",
      "Kind of hard to mess up a steak but they did.",
      "Although I very much liked the look and sound of this place, the actual experience was a bit disappointing.",
      "I just don't know how this place managed to served the blandest food I have ever eaten when they are preparing Indian cuisine.",
      "Worst service to boot, but that is the least of their worries.",
      "Service was fine and the waitress was friendly.",
      "The guys all had steaks, and our steak loving son who has had steak at the best and worst places said it was the best steak he's ever eaten.",
      "We thought you'd have to venture further away to get good sushi, but this place really hit the spot that night.",
      "Host staff were, for lack of a better word, BITCHES!",
      "Bland... Not a liking this place for a number of reasons and I don't want to waste time on bad reviewing.. I'll leave it at that...",
      "Phenomenal food, service and ambiance.",
      "I wouldn't return.",
      "Definitely worth venturing off the strip for the pork belly, will return next time I'm in Vegas.",
      "This place is way too overpriced for mediocre food.",
      "Penne vodka excellent!",
      "They have a good selection of food including a massive meatloaf sandwich, a crispy chicken wrap, a delish tuna melt and some tasty burgers.",
      "The management is rude.",
      "Delicious NYC bagels, good selections of cream cheese, real Lox with capers even.",
      "Great Subway, in fact it's so good when you come here every other Subway will not meet your expectations.",
      "I had a seriously solid breakfast here.",
      "This is one of the best bars with food in Vegas.",
      "He was extremely rude and really, there are so many other restaurants I would love to dine at during a weekend in Vegas.",
      "My drink was never empty and he made some really great menu suggestions.",
      "Don't do it!!!!",
      "The waiter wasn't helpful or friendly and rarely checked on us.",
      "My husband and I ate lunch here and were very disappointed with the food and service.",
      "And the red curry had so much bamboo shoots and wasn't very tasty to me.",
      "Nice blanket of moz over top but i feel like this was done to cover up the subpar food.",
      "The bathrooms are clean and the place itself is well decorated.",
      "The menu is always changing, food quality is going down & service is extremely slow.",
      "The service was a little slow , considering that were served by 3 people servers so the food was coming in a slow pace.",
      "I give it 2 thumbs down",
      "We watched our waiter pay a lot more attention to other tables and ignore us.",
      "My fiancé and I came in the middle of the day and we were greeted and seated right away.",
      "This is a great restaurant at the Mandalay Bay.",
      "We waited for forty five minutes in vain.",
      "Crostini that came with the salad was stale.",
      "Some highlights : Great quality nigiri here!",
      "the staff is friendly and the joint is always clean.",
      "this was a different cut than the piece the other day but still wonderful and tender s well as well flavored.",
      "I ordered the Voodoo pasta and it was the first time I'd had really excellent pasta since going gluten free several years ago.",
      "this place is good.",
      "Unfortunately, we must have hit the bakery on leftover day because everything we ordered was STALE.",
      "I came back today since they relocated and still not impressed.",
      "I was seated immediately.",
      "Their menu is diverse, and reasonably priced.",
      "Avoid at all cost!",
      "Restaurant is always full but never a wait.",
      "DELICIOUS!!",
      "This place is hands-down one of the best places to eat in the Phoenix metro area.",
      "So don't go there if you are looking for good food...",
      "I've never been treated so bad.",
      "Bacon is hella salty.",
      "We also ordered the spinach and avocado salad; the ingredients were sad and the dressing literally had zero taste.",
      "This really is how Vegas fine dining used to be, right down to the menus handed to the ladies that have no prices listed.",
      "The waitresses are very friendly.",
      "Lordy, the Khao Soi is a dish that is not to be missed for curry lovers!",
      "Everything on the menu is terrific and we were also thrilled that they made amazing accommodations for our vegetarian daughter.",
      "Perhaps I caught them on an off night judging by the other reviews, but I'm not inspired to go back.",
      "The service here leaves a lot to be desired.",
      "The atmosphere is modern and hip, while maintaining a touch of coziness.",
      "Not a weekly haunt, but definitely a place to come back to every once in a while.",
      "We literally sat there for 20 minutes with no one asking to take our order.",
      "The burger had absolutely no flavor - the meat itself was totally bland, the burger was overcooked and there was no charcoal flavor.",
      "I also decided not to send it back because our waitress looked like she was on the verge of having a heart attack.",
      "I dressed up to be treated so rudely!",
      "It was probably dirt.",
      "Love this place, hits the spot when I want something healthy but not lacking in quantity or flavor.",
      "I ordered the Lemon raspberry ice cocktail which was also incredible.",
      "The food sucked, which we expected but it sucked more than we could have imagined.",
      "Interesting decor.",
      "What I really like there is the crepe station.",
      "Also were served hot bread and butter, and home made potato chips with bacon bits on top....very original and very good.",
      "you can watch them preparing the delicious food!)",
      "Both of the egg rolls were fantastic.",
      "When my order arrived, one of the gyros was missing.",
      "I had a salad with the wings, and some ice cream for dessert and left feeling quite satisfied.",
      "I'm not really sure how Joey's was voted best hot dog in the Valley by readers of Phoenix Magazine.",
      "The best place to go for a tasty bowl of Pho!",
      "The live music on Fridays totally blows.",
      "I've never been more insulted or felt disrespected.",
      "Very friendly staff.",
      "It is worth the drive.",
      "I had heard good things about this place, but it exceeding every hope I could have dreamed of.",
      "Food was great and so was the serivce!",
      "The warm beer didn't help.",
      "Great brunch spot.",
      "Service is friendly and inviting.",
      "Very good lunch spot.",
      "I've lived here since 1979 and this was the first (and last) time I've stepped foot into this place.",
      "The WORST EXPERIENCE EVER.",
      "Must have been an off night at this place.",
      "The sides are delish - mixed mushrooms, yukon gold puree, white corn - beateous.",
      "If that bug never showed up I would have given a 4 for sure, but on the other side of the wall where this bug was climbing was the kitchen.",
      "For about 10 minutes, we we're waiting for her salad when we realized that it wasn't coming any time soon.",
      "My friend loved the salmon tartar.",
      "Won't go back.",
      "Extremely Tasty!",
      "Waitress was good though!",
      "Soggy and not good.",
      "The Jamaican mojitos are delicious.",
      "Which are small and not worth the price.",
      "- the food is rich so order accordingly.",
      "The shower area is outside so you can only rinse, not take a full shower, unless you don't mind being nude for everyone to see!",
      "The service was a bit lacking.",
      "Lobster Bisque, Bussell Sprouts, Risotto, Filet ALL needed salt and pepper..and of course there is none at the tables.",
      "Hopefully this bodes for them going out of business and someone who can cook can come in.",
      "It was either too cold, not enough flavor or just bad.",
      "I loved the bacon wrapped dates.",
      "This is an unbelievable BARGAIN!",
      "The folks at Otto always make us feel so welcome and special.",
      "As for the \"mains,\" also uninspired.",
      "This is the place where I first had pho and it was amazing!!",
      "This wonderful experience made this place a must-stop whenever we are in town again.",
      "If the food isn't bad enough for you, then enjoy dealing with the world's worst/annoying drunk people.",
      "Very very fun chef.",
      "Ordered a double cheeseburger & got a single patty that was falling apart (picture uploaded) Yeah, still sucks.",
      "Great place to have a couple drinks and watch any and all sporting events as the walls are covered with TV's.",
      "If it were possible to give them zero stars, they'd have it.",
      "The descriptions said \"yum yum sauce\" and another said \"eel sauce\", yet another said \"spicy mayo\"...well NONE of the rolls had sauces on them.",
      "I'd say that would be the hardest decision... Honestly, all of M's dishes taste how they are supposed to taste (amazing).",
      "If she had not rolled the eyes we may have stayed... Not sure if we will go back and try it again.",
      "Everyone is very attentive, providing excellent customer service.",
      "Horrible - don't waste your time and money.",
      "Now this dish was quite flavourful.",
      "By this time our side of the restaurant was almost empty so there was no excuse.",
      "(It wasn't busy either) Also, the building was FREEZING cold.",
      "like the other reviewer said \"you couldn't pay me to eat at this place again.\"",
      "-Drinks took close to 30 minutes to come out at one point.",
      "Seriously flavorful delights, folks.",
      "Much better than the other AYCE sushi place I went to in Vegas.",
      "The lighting is just dark enough to set the mood.",
      "Based on the sub-par service I received and no effort to show their gratitude for my business I won't be going back.",
      "Owner's are really great people.!",
      "There is nothing privileged about working/eating there.",
      "The Greek dressing was very creamy and flavorful.",
      "Overall, I don't think that I would take my parents to this place again because they made most of the similar complaints that I silently felt too.",
      "Now the pizza itself was good the peanut sauce was very tasty.",
      "We had 7 at our table and the service was pretty fast.",
      "Fantastic service here.",
      "I as well would've given godfathers zero stars if possible.",
      "They know how to make them here.",
      "very tough and very short on flavor!",
      "I hope this place sticks around.",
      "I have been in more than a few bars in Vegas, and do not ever recall being charged for tap water.",
      "The restaurant atmosphere was exquisite.",
      "Good service, very clean, and inexpensive, to boot!",
      "The seafood was fresh and generous in portion.",
      "Plus, it's only 8 bucks.",
      "The service was not up to par, either.",
      "Thus far, have only visited twice and the food was absolutely delicious each time.",
      "Just as good as when I had it more than a year ago!",
      "For a self proclaimed coffee cafe, I was wildly disappointed.",
      "The Veggitarian platter is out of this world!",
      "You cant go wrong with any of the food here.",
      "You can't beat that.",
      "Stopped by this place while in Madison for the Ironman, very friendly, kind staff.",
      "The chefs were friendly and did a good job.",
      "I've had better, not only from dedicated boba tea spots, but even from Jenni Pho.",
      "I liked the patio and the service was outstanding.",
      "The goat taco didn't skimp on the meat and wow what FLAVOR!",
      "I think not again",
      "I had the mac salad and it was pretty bland so I will not be getting that again.",
      "I went to Bachi Burger on a friend's recommendation and was not disappointed.",
      "Service stinks here!",
      "I waited and waited.",
      "This place is not quality sushi, it is not a quality restaurant.",
      "I would definitely recommend the wings as well as the pizza.",
      "Great Pizza and Salads!",
      "Things that went wrong: - They burned the saganaki.",
      "We waited an hour for what was a breakfast I could have done 100 times better at home.",
      "This place is amazing!",
      "I hate to disagree with my fellow Yelpers, but my husband and I were so disappointed with this place.",
      "Waited 2 hours & never got either of our pizzas as many other around us who came in later did!",
      "Just don't know why they were so slow.",
      "The staff is great, the food is delish, and they have an incredible beer selection.",
      "I live in the neighborhood so I am disappointed I won't be back here, because it is a convenient location.",
      "I didn't know pulled pork could be soooo delicious.",
      "You get incredibly fresh fish, prepared with care.",
      "Before I go in to why I gave a 1 star rating please know that this was my third time eating at Bachi burger before writing a review.",
      "I love the fact that everything on their menu is worth it.",
      "Never again will I be dining at this place!",
      "The food was excellent and service was very good.",
      "Good beer & drink selection and good food selection.",
      "Please stay away from the shrimp stir fried noodles.",
      "The potato chip order was sad... I could probably count how many chips were in that box and it was probably around 12.",
      "Food was really boring.",
      "Good Service-check!",
      "This greedy corporation will NEVER see another dime from me!",
      "Will never, ever go back.",
      "As much as I'd like to go back, I can't get passed the atrocious service and will never return.",
      "In the summer, you can dine in a charming outdoor patio - so very delightful.",
      "I did not expect this to be so good!",
      "Fantastic food!",
      "She ordered a toasted English muffin that came out untoasted.",
      "The food was very good.",
      "Never going back.",
      "Great food for the price, which is very high quality and house made.",
      "The bus boy on the other hand was so rude.",
      "By this point, my friends and I had basically figured out this place was a joke and didn't mind making it publicly and loudly known.",
      "Back to good BBQ, lighter fare, reasonable pricing and tell the public they are back to the old ways.",
      "And considering the two of us left there very full and happy for about $20, you just can't go wrong.",
      "All the bread is made in-house!",
      "The only downside is the service.",
      "Also, the fries are without a doubt the worst fries I've ever had.",
      "Service was exceptional and food was a good as all the reviews.",
      "A couple of months later, I returned and had an amazing meal.",
      "Favorite place in town for shawarrrrrrma!!!!!!",
      "The black eyed peas and sweet potatoes... UNREAL!",
      "You won't be disappointed.",
      "They could serve it with just the vinaigrette and it may make for a better overall dish, but it was still very good.",
      "I go to far too many places and I've never seen any restaurant that serves a 1 egg breakfast, especially for $4.00.",
      "When my mom and I got home she immediately got sick and she only had a few bites of salad.",
      "The servers are not pleasant to deal with and they don't always honor Pizza Hut coupons.",
      "Both of them were truly unbelievably good, and I am so glad we went back.",
      "We had fantastic service, and were pleased by the atmosphere.",
      "Everything was gross.",
      "I love this place.",
      "Great service and food.",
      "First - the bathrooms at this location were dirty- Seat covers were not replenished & just plain yucky!!!",
      "The burger... I got the \"Gold Standard\" a $17 burger and was kind of disappointed.",
      "OMG, the food was delicioso!",
      "There is nothing authentic about this place.",
      "the spaghetti is nothing special whatsoever.",
      "Of all the dishes, the salmon was the best, but all were great.",
      "The vegetables are so fresh and the sauce feels like authentic Thai.",
      "It's worth driving up from Tucson!",
      "The selection was probably the worst I've seen in Vegas.....there was none.",
      "Pretty good beer selection too.",
      "This place is like Chipotle, but BETTER.",
      "Classy/warm atmosphere, fun and fresh appetizers, succulent steaks (Baseball steak!!!!!",
      "5 stars for the brick oven bread app!",
      "I have eaten here multiple times, and each time the food was delicious.",
      "We sat another ten minutes and finally gave up and left.",
      "He was terrible!",
      "Everyone is treated equally special.",
      "It shouldn't take 30 min for pancakes and eggs.",
      "It was delicious!!!",
      "On the good side, the staff was genuinely pleasant and enthusiastic - a real treat.",
      "Sadly, Gordon Ramsey's Steak is a place we shall sharply avoid during our next trip to Vegas.",
      "As always the evening was wonderful and the food delicious!",
      "Best fish I've ever had in my life!",
      "(The bathroom is just next door and very nice.)",
      "The buffet is small and all the food they offered was BLAND.",
      "This is an Outstanding little restaurant with some of the Best Food I have ever tasted.",
      "Pretty cool I would say.",
      "Definitely a turn off for me & i doubt I'll be back unless someone else is buying.",
      "Server did a great job handling our large rowdy table.",
      "I find wasting food to be despicable, but this just wasn't food.",
      "My wife had the Lobster Bisque soup which was lukewarm.",
      "Would come back again if I had a sushi craving while in Vegas.",
      "The staff are great, the ambiance is great.",
      "He deserves 5 stars.",
      "I left with a stomach ache and felt sick the rest of the day.",
      "They dropped more than the ball.",
      "The dining space is tiny, but elegantly decorated and comfortable.",
      "They will customize your order any way you'd like, my usual is Eggplant with Green Bean stir fry, love it!",
      "And the beans and rice were mediocre at best.",
      "Best tacos in town by far!!",
      "I took back my money and got outta there.",
      "In an interesting part of town, this place is amazing.",
      "RUDE & INCONSIDERATE MANAGEMENT.",
      "The staff are now not as friendly, the wait times for being served are horrible, no one even says hi for the first 10 minutes.",
      "I won't be back.",
      "They have great dinners.",
      "The service was outshining & I definitely recommend the Halibut.",
      "The food was terrible.",
      "WILL NEVER EVER GO BACK AND HAVE TOLD MANY PEOPLE WHAT HAD HAPPENED.",
      "I don't recommend unless your car breaks down in front of it and you are starving.",
      "I will come back here every time I'm in Vegas.",
      "This place deserves one star and 90% has to do with the food.",
      "This is a disgrace.",
      "Def coming back to bowl next time",
      "If you want healthy authentic or ethic food, try this place.",
      "I will continue to come here on ladies night andddd date night ... highly recommend this place to anyone who is in the area (;",
      "I have been here several times in the past, and the experience has always been great.",
      "We walked away stuffed and happy about our first Vegas buffet experience.",
      "Service was excellent and prices are pretty reasonable considering this is Vegas and located inside the Crystals shopping mall by Aria.",
      "To summarize... the food was incredible, nay, transcendant... but nothing brings me joy quite like the memory of the pneumatic condiment dispenser.",
      "I'm probably one of the few people to ever go to Ians and not like it.",
      "Kids pizza is always a hit too with lots of great side dish options for the kiddos!",
      "Service is perfect and the family atmosphere is nice to see.",
      "Cooked to perfection and the service was impeccable.",
      "This one is simply a disappointment.",
      "Overall, I was very disappointed with the quality of food at Bouchon.",
      "I don't have to be an accountant to know I'm getting screwed!",
      "Great place to eat, reminds me of the little mom and pop shops in the San Francisco Bay Area.",
      "Today was my first taste of a Buldogis Gourmet Hot Dog and I have to tell you it was more than I ever thought possible.",
      "Left very frustrated.",
      "I'll definitely be in soon again.",
      "Food was really good and I got full petty fast.",
      "Service was fantastic.",
      "TOTAL WASTE OF TIME.",
      "I don't know what kind it is but they have the best iced tea.",
      "Come hungry, leave happy and stuffed!",
      "For service, I give them no stars.",
      "I can assure you that you won't be disappointed.",
      "I can take a little bad service but the food sucks.",
      "Gave up trying to eat any of the crust (teeth still sore).",
      "But now I was completely grossed out.",
      "I really enjoyed eating here.",
      "First time going but I think I will quickly become a regular.",
      "Our server was very nice, and even though he looked a little overwhelmed with all of our needs, he stayed professional and friendly until the end.",
      "From what my dinner companions told me...everything was very fresh with nice texture and taste.",
      "On the ground, right next to our table was a large, smeared, been-stepped-in-and-tracked-everywhere pile of green bird poop.",
      "Furthermore, you can't even find hours of operation on the website!",
      "We've tried to like this place but after 10+ times I think we're done with them.",
      "What a mistake that was!",
      "No complaints!",
      "This is some seriously good pizza and I'm an expert/connisseur on the topic.",
      "Waiter was a jerk.",
      "Strike 2, who wants to be rushed.",
      "These are the nicest restaurant owners I've ever come across.",
      "I never come again.",
      "We loved the biscuits!!!",
      "Service is quick and friendly.",
      "Ordered an appetizer and took 40 minutes and then the pizza another 10 minutes.",
      "So absolutley fantastic.",
      "It was a huge awkward 1.5lb piece of cow that was 3/4ths gristle and fat.",
      "definitely will come back here again.",
      "I like Steiners because it's dark and it feels like a bar.",
      "Wow very spicy but delicious.",
      "If you're not familiar, check it out.",
      "I'll take my business dinner dollars elsewhere.",
      "I'd love to go back.",
      "Anyway, this FS restaurant has a wonderful breakfast/lunch.",
      "Nothing special.",
      "Each day of the week they have a different deal and it's all so delicious!",
      "Not to mention the combination of pears, almonds and bacon is a big winner!",
      "Will not be back.",
      "Sauce was tasteless.",
      "The food is delicious and just spicy enough, so be sure to ask for spicier if you prefer it that way.",
      "My ribeye steak was cooked perfectly and had great mesquite flavor.",
      "I don't think we'll be going back anytime soon.",
      "Food was so gooodd.",
      "I am far from a sushi connoisseur but I can definitely tell the difference between good food and bad food and this was certainly bad food.",
      "I was so insulted.",
      "The last 3 times I had lunch here has been bad.",
      "The chicken wings contained the driest chicken meat I have ever eaten.",
      "The food was very good and I enjoyed every mouthful, an enjoyable relaxed venue for couples small family groups etc.",
      "Nargile - I think you are great.",
      "Best tater tots in the southwest.",
      "We loved the place.",
      "Definitely not worth the $3 I paid.",
      "The vanilla ice cream was creamy and smooth while the profiterole (choux) pastry was fresh enough.",
      "Im in AZ all the time and now have my new spot.",
      "The manager was the worst.",
      "The inside is really quite nice and very clean.",
      "The food was outstanding and the prices were very reasonable.",
      "I don't think I'll be running back to Carly's anytime soon for food.",
      "This is was due to the fact that it took 20 minutes to be acknowledged, then another 35 minutes to get our food...and they kept forgetting things.",
      "Love the margaritas, too!",
      "This was my first and only Vegas buffet and it did not disappoint.",
      "Very good, though!",
      "The one down note is the ventilation could use some upgrading.",
      "Great pork sandwich.",
      "Don't waste your time here.",
      "Total letdown, I would much rather just go to the Camelback Flower Shop and Cartel Coffee.",
      "Third, the cheese on my friend's burger was cold.",
      "We enjoy their pizza and brunch.",
      "The steaks are all well trimmed and also perfectly cooked.",
      "We had a group of 70+ when we claimed we would only have 40 and they handled us beautifully.",
      "I LOVED it!",
      "We asked for the bill to leave without eating and they didn't bring that either.",
      "This place is a jewel in Las Vegas, and exactly what I've been hoping to find in nearly ten years living here.",
      "Seafood was limited to boiled shrimp and crab legs but the crab legs definitely did not taste fresh.",
      "The selection of food was not the best.",
      "Delicious and I will absolutely be back!",
      "This isn't a small family restaurant, this is a fine dining establishment.",
      "They had a toro tartare with a cavier that was extraordinary and I liked the thinly sliced wagyu with white truffle.",
      "I dont think I will be back for a very long time.",
      "It was attached to a gas station, and that is rarely a good sign.",
      "How awesome is that.",
      "I will be back many times soon.",
      "The menu had so much good stuff on it i could not decide!",
      "Worse of all, he humiliated his worker right in front of me..Bunch of horrible name callings.",
      "CONCLUSION: Very filling meals.",
      "Their daily specials are always a hit with my group.",
      "And then tragedy struck.",
      "The pancake was also really good and pretty large at that.",
      "This was my first crawfish experience, and it was delicious!",
      "Their monster chicken fried steak and eggs is my all time favorite.",
      };

      labels = {
      1,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      1,
      1,
      1,
      0,
      0,
      1,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      1,
      0,
      0,
      1,
      0,
      1,
      0,
      1,
      1,
      1,
      0,
      1,
      0,
      1,
      0,
      0,
      1,
      0,
      1,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      1,
      1,
      0,
      0,
      1,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      1,
      0,
      1,
      0,
      1,
      1,
      1,
      0,
      1,
      0,
      1,
      0,
      0,
      1,
      1,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      0,
      0,
      1,
      0,
      1,
      0,
      1,
      1,
      0,
      1,
      1,
      1,
      1,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      1,
      1,
      0,
      1,
      1,
      1,
      0,
      0,
      1,
      0,
      1,
      1,
      1,
      1,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      1,
      0,
      1,
      0,
      0,
      1,
      1,
      1,
      1,
      0,
      1,
      1,
      1,
      0,
      0,
      0,
      1,
      0,
      0,
      1,
      0,
      1,
      1,
      0,
      1,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      0,
      1,
      1,
      0,
      1,
      0,
      1,
      0,
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      1,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      1,
      1,
      1,
      0,
      1,
      1,
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      1,
      1,
      0,
      0,
      1,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      1,
      0,
      0,
      0,
      1,
      0,
      1,
      1,
      0,
      1,
      0,
      1,
      1,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      1,
      1,
      1,
      0,
      1,
      0,
      1,
      0,
      0,
      1,
      1,
      1,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      1,
      1,
      0,
      1,
      1,
      0,
      0,
      1,
      0,
      0,
      1,
      1,
      1,
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      1,
      0,
      1,
      1,
      0,
      1,
      1,
      1,
      0,
      1,
      1,
      0,
      1,
      0,
      0,
      1,
      1,
      1,
      0,
      0,
      1,
      1,
      0,
      1,
      0,
      1,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      0,
      1,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      1,
      1,
      1,
      0,
      0,
      0,
      1,
      1,
      0,
      1,
      1,
      1,
      0,
      1,
      1,
      0,
      1,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      1,
      1,
      0,
      0,
      1,
      0,
      1,
      1,
      0,
      1,
      0,
      1,
      1,
      1,
      1,
      0,
      1,
      1,
      0,
      1,
      1,
      0,
      0,
      1,
      1,
      0,
      1,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      0,
      1,
      1,
      0,
      1,
      1,
      0,
      0,
      1,
      1,
      1,
      0,
      1,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      0,
      1,
      0,
      0,
      1,
      1,
      1,
      0,
      0,
      1,
      1,
      1,
      0,
      1,
      1,
      0,
      1,
      1,
      1
      };
   }
};