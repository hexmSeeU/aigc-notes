const subpath = process.env.BLOG_TEST_SUBPATH === '1';
const port = subpath ? 1314 : 1313;
const baseURL = `http://127.0.0.1:${port}/${subpath ? 'aigc-notes/' : ''}`;
export default {testDir:'./tests',use:{baseURL},webServer:{command:`node scripts/prepare-test-fixtures.mjs && hugo server --buildDrafts --bind 127.0.0.1 --port ${port} --baseURL ${baseURL} --appendPort=false`,url:baseURL,reuseExistingServer:!process.env.CI},reporter:'list'};
